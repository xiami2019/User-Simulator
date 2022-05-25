# training pptod-small
import os
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from itertools import chain
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.nn.utils.rnn import pad_sequence

from evaluator import MultiWozEvaluator
from reader import MultiWOZReader, MultiWOZIterator
from utils.utils import get_or_create_logger, save_json
from utils import definitions
from runner import BaseRunner

logger = get_or_create_logger(__name__)

class MultiWOZDatasetPPTOD(Dataset):
    def __init__(self, original_data, data_type, tokenizer, bs_prefix_id, da_prefix_id, nlg_prefix_id, sos_context_token_id, eos_context_token_id) -> None:
        super().__init__()
        self.data_type = data_type
        self.tokenizer = tokenizer

        self.bs_prefix_id = bs_prefix_id
        self.da_prefix_id = da_prefix_id
        self.nlg_prefix_id = nlg_prefix_id
        self.sos_context_token_id = sos_context_token_id
        self.eos_context_token_id = eos_context_token_id 

        self.data = self.construct_data(original_data)

    def constraint_history_length(self, dialog_history, additional_token_num=2):
        context = dialog_history[:]
        context_len = sum([len(t) for t in context]) + additional_token_num
        while context_len > self.tokenizer.model_max_length:
            context_len -= len(context[0])
            context.pop(0)
        context = list(chain(*context))

        return context

    def construct_data(self, original_data):
        '''
        transform session data into turn data
        '''
        data_list = []
        for dial in original_data:
            dialog_history = []
            for turn in dial:
                dialog_history.append(turn['user'])
                
                context_for_bs = self.constraint_history_length(dialog_history, 2 + len(self.bs_prefix_id))
                context_for_da = self.constraint_history_length(dialog_history, 2 + len(self.da_prefix_id) + len(turn['dbpn']))
                context_for_nlg = self.constraint_history_length(dialog_history, 2 + len(self.nlg_prefix_id) + len(turn['dbpn']))
                bs_input = self.bs_prefix_id + [self.sos_context_token_id] + context_for_bs + [self.eos_context_token_id]
                bs_output = turn['bspn'] + [self.tokenizer.eos_token_id]
                da_input = self.da_prefix_id + [self.sos_context_token_id] + context_for_da + [self.eos_context_token_id] + turn['dbpn']
                da_output = turn['aspn'] + [self.tokenizer.eos_token_id]
                nlg_input = self.nlg_prefix_id + [self.sos_context_token_id] + context_for_nlg + [self.eos_context_token_id] + turn['dbpn']
                nlg_output = turn['redx'] + [self.tokenizer.eos_token_id]

                dialog_history.append(turn['redx'])

                data_list.extend([(bs_input, bs_output), (da_input, da_output), (nlg_input, nlg_output)])
        
        return data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1]

class PPTODReader(MultiWOZReader):
    def __init__(self, cfg, version):
        super().__init__(cfg, version)

    def init_tokenizer(self):
        if self.cfg.ckpt is not None:
            return T5Tokenizer.from_pretrained(self.cfg.ckpt)
        elif self.cfg.train_from is not None:
            return T5Tokenizer.from_pretrained(self.cfg.train_from)
        else:
            tokenizer = T5Tokenizer.from_pretrained(self.cfg.backbone)

        special_tokens = []

        # add domains
        domains = definitions.ALL_DOMAINS + ["general"]
        for domain in sorted(domains):
            token = "[" + domain + "]"
            special_tokens.append(token)

        # add intents
        intents = list(set(chain(*definitions.DIALOG_ACTS.values())))
        for intent in sorted(intents):
            token = "[" + intent + "]"
            special_tokens.append(token)

        intents = list(set(chain(*definitions.USER_ACTS.values())))
        for intent in sorted(intents):
            token = "[" + intent + "]"
            special_tokens.append(token)

        # add slots
        slots = list(set(definitions.ALL_INFSLOT + definitions.ALL_REQSLOT))

        for slot in sorted(slots):
            token = "[value_" + slot + "]"
            special_tokens.append(token)

        special_tokens.extend(definitions.SPECIAL_TOKENS)
        special_tokens.extend(['<sos_context>', '<eos_context>'])

        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

        return tokenizer

class CollateForPPTOD(object):
    def __init__(self, pad_id) -> None:
        self.pad_id = pad_id

    def __call__(self, batch):
        batch_input_ids = []
        batch_label_ids = []
        for i in batch:
            batch_input_ids.append(i[0])
            batch_label_ids.append(i[1])

        batch_input_tensor = [torch.tensor(i, dtype=torch.long) for i in batch_input_ids]
        batch_label_tensor = [torch.tensor(i, dtype=torch.long) for i in batch_label_ids]
        batch_input_tensor = pad_sequence(batch_input_tensor, batch_first=True, padding_value=self.pad_id)
        batch_label_tensor = pad_sequence(batch_label_tensor, batch_first=True, padding_value=self.pad_id)

        return batch_input_tensor, batch_label_tensor

class PPTODRunner(BaseRunner):
    def __init__(self, cfg, reader):
        super().__init__(cfg, reader)
        self.cfg = cfg
        self.reader = reader
        self.iterator = MultiWOZIterator(reader)
        self.evaluator = MultiWozEvaluator(reader, cfg.pred_data_type)

        bs_prefix_text = 'translate dialogue to belief state:'
        da_prefix_text = 'translate dialogue to dialogue action:'
        nlg_prefix_text = 'translate dialogue to system response:'
        self.bs_prefix_id = self.reader.tokenizer.convert_tokens_to_ids(self.reader.tokenizer.tokenize(bs_prefix_text))
        self.da_prefix_id = self.reader.tokenizer.convert_tokens_to_ids(self.reader.tokenizer.tokenize(da_prefix_text))
        self.nlg_prefix_id = self.reader.tokenizer.convert_tokens_to_ids(self.reader.tokenizer.tokenize(nlg_prefix_text))
        self.sos_context_token_id = self.reader.tokenizer.convert_tokens_to_ids(['<sos_context>'])[0]
        self.eos_context_token_id = self.reader.tokenizer.convert_tokens_to_ids(['<eos_context>'])[0]

    def load_model(self):
        if self.cfg.ckpt is not None:
            model_path = self.cfg.ckpt
        elif self.cfg.train_from is not None:
            model_path = self.cfg.train_from
        else:
            model_path = self.cfg.backbone

        model = T5ForConditionalGeneration.from_pretrained(model_path)

        logger.info("Load model from {}".format(model_path))

        model.resize_token_embeddings(self.reader.vocab_size)
        model.to(self.cfg.device)
        return model

    def train(self):
        collate_fn = CollateForPPTOD(self.reader.pad_token_id)
        train_dataset = MultiWOZDatasetPPTOD(self.reader.data['train'], 'train', self.reader.tokenizer, self.bs_prefix_id, self.da_prefix_id, self.nlg_prefix_id, self.sos_context_token_id, self.eos_context_token_id)
        train_dataloader = DataLoader(train_dataset, batch_size=self.cfg.batch_size, shuffle=True, collate_fn=collate_fn)
        num_training_steps_per_epoch = len(train_dataloader)
        optimizer, scheduler = self.get_optimizer_and_scheduler(num_training_steps_per_epoch, self.cfg.batch_size)

        best_combined_score = 0.0
        best_epoch=0

        for epoch in range(1, self.cfg.epochs + 1):
            self.model.train()
            self.model.zero_grad()
            training_avg_loss = 0

            for step, batch in enumerate(tqdm(train_dataloader, desc='Epoch {} Training'.format(epoch))):
                inputs_ids, outputs_ids = batch
                inputs_ids = inputs_ids.to(self.cfg.device)
                outputs_ids = outputs_ids.to(self.cfg.device)
                attention_mask = torch.where(inputs_ids == self.reader.pad_token_id, 0, 1)

                model_outputs = self.model(
                    input_ids=inputs_ids,
                    attention_mask=attention_mask,
                    labels=outputs_ids,
                )

                loss = model_outputs.loss

                if self.cfg.grad_accum_steps > 1:
                    loss = loss / self.cfg.grad_accum_steps
                
                training_avg_loss += loss.item()
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)

                if (step + 1) % self.cfg.grad_accum_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
            
            logger.info("done {}/{} epoch; Average training loss: {}".format(epoch, self.cfg.epochs, training_avg_loss / len(train_dataloader)))

            if epoch > self.cfg.test_after_epochs:
                bleu, success, match = self.predict(predict_when_training=True)
                score = 0.5 * (success + match) + bleu
                logger.info('Epoch %d: match: %2.2f; success: %2.2f; bleu: %2.2f; score: %.2f' % (
                    epoch, match, success, bleu, score))

                if score > best_combined_score:
                    best_combined_score = score
                    best_epoch = epoch
                    self.save_model(epoch)

                logger.info('Best combined score: {} at epoch {}.'.format(best_combined_score, best_epoch))


    def predict(self, predict_when_training=False):
        self.model.eval()
        pred_batches, _, _, _ = self.iterator.get_batches(self.cfg.pred_data_type, self.cfg.batch_size, num_gpus=1)

        results = {}
        for dial_batch in tqdm(pred_batches, total=len(pred_batches), desc="Prediction"):
            batch_size = len(dial_batch)
            dial_history = [[] for _ in range(batch_size)]
            for turn_batch in self.iterator.transpose_batch(dial_batch):
                batch_bs_encoder_input_ids = []
                batch_da_encoder_input_ids = []
                batch_nlg_encoder_input_ids = []
                for t, turn in enumerate(turn_batch):
                    context_for_bs = self.iterator.flatten_dial_history(dial_history[t], len(self.bs_prefix_id) + len(turn['user']) + 1)
                    bs_encoder_input_ids = self.bs_prefix_id + [self.sos_context_token_id] + context_for_bs + turn['user'] + [self.eos_context_token_id]
                    batch_bs_encoder_input_ids.append(self.iterator.tensorize(bs_encoder_input_ids))
                    
                batch_bs_encoder_input_ids = pad_sequence(batch_bs_encoder_input_ids, batch_first=True, padding_value=self.reader.pad_token_id)
                batch_bs_encoder_input_ids = batch_bs_encoder_input_ids.to(self.cfg.device)
                attention_mask = torch.where(batch_bs_encoder_input_ids == self.reader.pad_token_id, 0, 1)
                
                # belief tracking
                with torch.no_grad():
                    belief_outputs = self.model.generate(input_ids=batch_bs_encoder_input_ids,
                                                        attention_mask=attention_mask,
                                                        eos_token_id=self.reader.eos_token_id,
                                                        max_length=100)
                belief_outputs = belief_outputs.cpu().numpy().tolist()
                decoded_belief_outputs = self.finalize_outputs(
                    belief_outputs, 'bspn_gen', '<eos_b>')

                for t, turn in enumerate(turn_batch):
                    turn.update(**decoded_belief_outputs[t])

                dbpn = []
                for turn in turn_batch:
                    bspn_gen = turn["bspn_gen"]
                    bspn_gen = self.reader.tokenizer.decode(
                        bspn_gen, clean_up_tokenization_spaces=False)

                    db_token = self.reader.bspn_to_db_pointer(bspn_gen, turn["turn_domain"])

                    dbpn_gen = self.reader.encode_text(
                        db_token,
                        bos_token='<sos_d>',
                        eos_token='<eos_d>')

                    turn["dbpn_gen"] = dbpn_gen

                    dbpn.append(dbpn_gen)

                for t, turn in enumerate(turn_batch):
                    context_for_da = self.iterator.flatten_dial_history(dial_history[t], len(self.da_prefix_id) + len(turn['user']) + len(turn["dbpn_gen"]) + 1)
                    context_for_nlg = self.iterator.flatten_dial_history(dial_history[t], len(self.nlg_prefix_id) + len(turn['user']) + len(turn["dbpn_gen"]) + 1)
                    
                    da_encoder_input_ids = self.da_prefix_id + [self.sos_context_token_id] + context_for_da + turn['user'] + [self.eos_context_token_id] + turn["dbpn_gen"]
                    nlg_encoder_input_ids = self.nlg_prefix_id + [self.sos_context_token_id] + context_for_nlg + turn['user'] + [self.eos_context_token_id] + turn["dbpn_gen"]

                    batch_da_encoder_input_ids.append(self.iterator.tensorize(da_encoder_input_ids))
                    batch_nlg_encoder_input_ids.append(self.iterator.tensorize(nlg_encoder_input_ids))

                    turn_domain = turn['turn_domain'][-1]
                    if "[" in turn_domain:
                        turn_domain = turn_domain[1:-1]

                batch_da_encoder_input_ids = pad_sequence(batch_da_encoder_input_ids,
                                                            batch_first=True,
                                                            padding_value=self.reader.pad_token_id)
                batch_nlg_encoder_input_ids = pad_sequence(batch_nlg_encoder_input_ids,
                                                            batch_first=True,
                                                            padding_value=self.reader.pad_token_id)

                batch_da_encoder_input_ids = batch_da_encoder_input_ids.to(self.cfg.device)
                batch_nlg_encoder_input_ids = batch_nlg_encoder_input_ids.to(self.cfg.device)

                attention_mask_da =  torch.where(
                    batch_da_encoder_input_ids == self.reader.pad_token_id, 0, 1)
                attention_mask_nlg = torch.where(
                    batch_nlg_encoder_input_ids == self.reader.pad_token_id, 0, 1)

                # action prediction
                with torch.no_grad():
                    aspn_outputs = self.model.generate(
                        input_ids=batch_da_encoder_input_ids,
                        attention_mask=attention_mask_da,
                        eos_token_id=self.reader.eos_token_id,
                        max_length=100,
                    )

                    resp_outputs = self.model.generate(
                        input_ids=batch_nlg_encoder_input_ids,
                        attention_mask=attention_mask_nlg,
                        eos_token_id=self.reader.eos_token_id,
                        max_length=200,
                    )

                aspn_outputs = aspn_outputs.cpu().numpy().tolist()
                resp_outputs = resp_outputs.cpu().numpy().tolist()
                decoded_action_outputs = self.finalize_outputs(aspn_outputs, 'aspn_gen', '<eos_a>')
                decoded_response_outputs = self.finalize_outputs(resp_outputs, 'resp_gen', '<eos_r>')

                for t, turn in enumerate(turn_batch):
                    turn.update(**decoded_action_outputs[t])
                    turn.update(**decoded_response_outputs[t])

                # update dial_hitory
                for t, turn in enumerate(turn_batch):
                    pv_text = turn['user'] + turn['resp_gen']
                    dial_history[t].append(pv_text)
                
            result = self.iterator.get_readable_batch(dial_batch)
            results.update(**result)      
        
        if predict_when_training == False:
            save_json(results, os.path.join(self.cfg.ckpt, self.cfg.output))

        bleu, success, match = self.evaluator.e2e_eval(results)

        return bleu, success, match


    def finalize_outputs(self, outputs, output_type, eos_token):
        '''
        output_type: bspn_gen, aspn_gen, resp_gen
        '''
        eos_token_id = self.reader.get_token_id(eos_token)
        batch_decoded = []
        for i, belief_output in enumerate(outputs):
            if belief_output[0] == self.reader.pad_token_id:
                belief_output = belief_output[1:]

            if eos_token_id not in belief_output:
                eos_idx = len(belief_output) - 1
            else:
                eos_idx = belief_output.index(eos_token_id)

            decoded = {}
            decoded[output_type] = belief_output[:eos_idx + 1]

            batch_decoded.append(decoded)

        return batch_decoded

def parse_config():
    parser = argparse.ArgumentParser()
    # dataset configuration
    parser.add_argument('--data_path_prefix', type=str, help='The path where the data stores.')
    parser.add_argument('--shuffle_mode', type=str, default='shuffle_session_level', 
        help="shuffle_session_level or shuffle_turn_level, it controls how we shuffle the training data.")
    parser.add_argument('--use_db_as_input', type=str, default='True', 
        help="True or False, whether includes db result as part of the input when generating response.")
    parser.add_argument('--add_special_decoder_token', default='True', type=str, help='Whether we discriminate the decoder start and end token for different tasks.')
    parser.add_argument('--train_data_ratio', type=float, default=1.0, help='the ratio of training data used for training the model')
    parser.add_argument("--version", type=str, default="2.0", choices=["2.0", "2.1"])

    # model configuration
    parser.add_argument('--backbone', type=str, default='pptod_small', help='pptod_small, pptod_base, pptod_large')
    parser.add_argument('--ckpt', type=str, default=None, help='the path that stores pretrained checkpoint.')
    parser.add_argument('--train_from', type=str, default=None)
    parser.add_argument("--model_name", type=str, default='pptod', help = 'mttod, pptod, ubar, galaxy')

    # training configuration
    parser.add_argument('--run_type', type=str, default='train', choices=['train', 'predict'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup_steps", type=int, default=-1)
    parser.add_argument("--warmup_ratio", type=float, default=0.2)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_to_keep_ckpt", type=int, default=1) 
    parser.add_argument("--model_dir", type=str, default='pptod_small_finetune', help="directory to save the model parameters.")
    parser.add_argument("--pred_data_type", type=str, default='test', choices=['test', 'dev'])
    parser.add_argument("--output", type=str, default='inference.json', help="generated results")
    parser.add_argument("--test_after_epochs", type=int, default=5)
    parser.add_argument("--no_validation", action="store_true")
    parser.add_argument("--no_learning_rate_decay", action="store_true")
    
    return parser.parse_args()

if __name__ == '__main__':
    if torch.cuda.is_available():
        logger.info('Cuda is available.')
    cuda_available = torch.cuda.is_available()
    multi_gpu_training = False
    if cuda_available:
        if torch.cuda.device_count() > 1:
            multi_gpu_training = True
            logger.info('Using Multi-GPU training, number of GPU is {}'.format(torch.cuda.device_count()))
        else:
            logger.info('Using single GPU training.')
    else:
        pass
    cfg = parse_config()
    device = torch.device('cuda')
    setattr(cfg, "device", device)

    if cfg.seed > 0:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

        logger.info("Set random seed to %d", cfg.seed)

    pptod_reader = MultiWOZReader(cfg, cfg.version)
    pptod_runner = PPTODRunner(cfg, pptod_reader)

    if cfg.run_type == 'train':
        pptod_runner.train()
    elif cfg.run_type == 'predict':
        bleu, success, match = pptod_runner.predict(predict_when_training=False)
        score = 0.5 * (success + match) + bleu
        logger.info('match: %2.2f; success: %2.2f; bleu: %2.2f; score: %.2f' % (
                    match, success, bleu, score))

    
