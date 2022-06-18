import os
import spacy
import torch
import random
import argparse
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from itertools import chain
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup

from evaluator import MultiWozEvaluator
from reader import MultiWOZIterator
from utils.utils import get_or_create_logger, save_json, load_json, save_pickle, load_pickle
from utils import definitions

from galaxy.models.model_base import ModelBase
from galaxy.models.generator import Generator
from galaxy.args import parse_args

from external_knowledges import MultiWozDB

logger = get_or_create_logger(__name__)

def clean_string(string):
    replace_mp = {
        " - ": "-",
        " ' ": "'",
        " n't": "n't",
        " 'm": "'m",
        " do not": " don't",
        " 's": "'s",
        " 've": "'ve",
        " 're": "'re"
    }
    for k, v in replace_mp.items():
        string = string.replace(k, v)
    return string

class Tokenizer(object):
    def __init__(self, vocab_path, special_tokens=[], tokenizer_type="Bert"):
        self.tokenizer_type = tokenizer_type
        self.nlp = spacy.load("en_core_web_sm")
        if tokenizer_type == "Bert":
            self.spec_convert_dict = {"[BOS]": "[unused0]", "[EOS]": "[unused1]"}
            for token in special_tokens:
                if token not in self.spec_convert_dict and token not in ['[PAD]', '[UNK]']:
                    self.spec_convert_dict[token] = f"[unused{len(self.spec_convert_dict)}]"
            self.spec_revert_dict = {v: k for k,
                                     v in self.spec_convert_dict.items()}
            special_tokens = [self.spec_convert_dict.get(tok, tok)
                              for tok in special_tokens]
            self.special_tokens = ("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")
            self.special_tokens += tuple(x for x in special_tokens if x not in self.special_tokens)
            # for x in special_tokens:
            #     if x not in self.special_tokens:
            #         self.special_tokens += (x,)

            self._tokenizer = BertTokenizer(vocab_path, never_split=self.special_tokens)
            for tok in self.special_tokens:
                assert tok in self._tokenizer.vocab, f"special token '{tok}' is not in the vocabulary"
            self.vocab_size = len(self._tokenizer.vocab)
        # elif tokenizer_type == "GPT2":
        #     self.spec_convert_dict = {"[UNK]": "<unk>"}
        #     self.spec_revert_dict = {v: k for k,
        #                              v in self.spec_convert_dict.items()}
        #     special_tokens = [tok for tok in special_tokens
        #                       if tok not in self.spec_convert_dict]
        #     vocab_file = os.path.join(vocab_path, "vocab.json")
        #     merges_file = os.path.join(vocab_path, "merges.txt")
        #     self._tokenizer = GPT2Tokenizer(vocab_file, merges_file, special_tokens=special_tokens)
        #     self.num_specials = len(special_tokens)
        #     self.vocab_size = len(self._tokenizer)
        else:
            raise ValueError

    def tokenize(self, text):
        text = ' '.join([self.spec_convert_dict.get(tok, tok) for tok in text.split()])
        return self._tokenizer.tokenize(text)

    def convert_tokens_to_ids(self, tokens):
        if self.tokenizer_type == "Bert":
            tokens = [self.spec_convert_dict.get(tok, tok) for tok in tokens]
            ids = self._tokenizer.convert_tokens_to_ids(tokens)
            return ids
        else:
            tokens = [self.spec_convert_dict.get(tok, tok) for tok in tokens]
            ids = self._tokenizer.convert_tokens_to_ids(tokens)
            ids = [(i + self.num_specials) % self.vocab_size for i in ids]
            return ids

    def convert_ids_to_tokens(self, ids):
        if self.tokenizer_type == "Bert":
            tokens = self._tokenizer.convert_ids_to_tokens(ids)
            tokens = [self.spec_revert_dict.get(tok, tok) for tok in tokens]
            return tokens
        else:
            ids = [(i - self.num_specials) % self.vocab_size for i in ids]
            tokens = self._tokenizer.convert_ids_to_tokens(ids)
            tokens = [self.spec_revert_dict.get(tok, tok) for tok in tokens]
            return tokens

    def decode(self, ids, ignore_tokens=[], clean_up_tokenization_spaces=False):
        # clean_up_tokenization_spaces=False 没用，纯粹为了兼容别的代码
        tokens = self.convert_ids_to_tokens(ids)
        if len(ignore_tokens) > 0:
            ignore_tokens = set(ignore_tokens)
            tokens = [tok for tok in tokens if tok not in ignore_tokens]
        if self.tokenizer_type == "Bert":
            string = " ".join(tokens).replace(" ##", "")
        else:
            string = "".join(tokens)
            string = bytearray([self._tokenizer.byte_decoder[c]
                                for c in string]).decode("utf-8")
        # string = clean_string(string)
        return string

    def __len__(self):
        return len(self._tokenizer)

class MultiWOZDatasetGalaxy(Dataset):
    def __init__(self, original_data, data_type, tokenizer) -> None:
        super().__init__()
        self.data_type = data_type
        self.tokenizer = tokenizer

        self.data = self.construct_data(original_data)

    def constraint_history_length(self, dialog_history, additional_token_num=2):
        context = dialog_history[:]
        # context_len = sum([len(t) for t in context]) + additional_token_num
        # while context_len > self.tokenizer.model_max_length:
        #     context_len -= len(context[0])
        #     context.pop(0)
        # context = list(chain(*context))

        return context

    def construct_data(self, original_data):
        '''
        transform session data into turn data
        '''
        data_list = []
        for dial in original_data:
            dialog_history = []
            for turn in dial:
                context = self.constraint_history_length(dialog_history, len(turn['user']))
                input_ids = context + [turn['user']]
                output_ids = turn['bspn'] + turn['dbpn'] + turn['aspn'] + turn['redx']
                data_list.append({'src': input_ids, 'tgt': output_ids, 'act': turn['act']})
                dialog_history.append(turn['user'])
                dialog_history.append(turn['bspn'] + turn['dbpn'] + turn['aspn'] + turn['redx'])

        return data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class CollateForGalaxy(object):
    def __init__(self, pad_id, max_len, max_ctx_turn) -> None:
        self.pad_id = pad_id
        self.max_len = max_len
        self.max_ctx_turn = max_ctx_turn
        self.sys_id = 0
        self.usr_id = 1
    
    def max_lens(self, X):
        lens = [len(X)]
        while isinstance(X[0], list):
            lens.append(max(map(len, X)))
            X = [x for xs in X for x in xs]
        return lens

    def list2np(self, X, padding=0, dtype="int64"):
        shape = self.max_lens(X)
        ret = np.full(shape, padding, dtype=np.int32)

        if len(shape) == 1:
            ret = np.array(X)
        elif len(shape) == 2:
            for i, x in enumerate(X):
                ret[i, :len(x)] = np.array(x)
        elif len(shape) == 3:
            for i, xs in enumerate(X):
                for j, x in enumerate(xs):
                    ret[i, j, :len(x)] = np.array(x)
        return ret.astype(dtype)

    def __call__(self, samples):
        batch_size = len(samples)
        src = [sp["src"][-self.max_ctx_turn:] for sp in samples]
        src_token, src_pos, src_turn, src_role = [], [], [], []
        for utts in src:
            utt_lens = [len(utt) for utt in utts]

            # Token ids
            src_token.append(list(chain(*utts))[-self.max_len:])

            # Position ids
            pos = [list(range(l)) for l in utt_lens]
            src_pos.append(list(chain(*pos))[-self.max_len:])

            # Turn ids
            turn = [[len(utts) - i] * l for i, l in enumerate(utt_lens)]
            src_turn.append(list(chain(*turn))[-self.max_len:])

            # Role ids
            role = [[self.sys_id if (len(utts) - i) % 2 == 0 else self.usr_id] * l for i, l in enumerate(utt_lens)]
            src_role.append(list(chain(*role))[-self.max_len:])
        
        src_token = self.list2np(src_token, padding=self.pad_id)
        src_pos = self.list2np(src_pos, padding=self.pad_id)
        src_turn = self.list2np(src_turn, padding=self.pad_id)
        src_role = self.list2np(src_role, padding=self.pad_id)

        batch = {}
        batch['src_token'] = src_token
        batch['src_mask'] = (src_token != self.pad_id).astype("int64")
        batch["src_pos"] = src_pos
        batch["src_type"] = src_role
        batch["src_turn"] = src_turn

        if "tgt" in samples[0]:
            tgt = [sp["tgt"] for sp in samples]

            # Token ids & Label ids
            tgt_token = self.list2np(tgt, padding=self.pad_id)

            # Position ids
            tgt_pos = np.zeros_like(tgt_token)
            tgt_pos[:] = np.arange(tgt_token.shape[1], dtype=tgt_token.dtype)

            # Turn ids
            tgt_turn = np.zeros_like(tgt_token)

            # Role ids
            tgt_role = np.full_like(tgt_token, self.sys_id)

            batch["tgt_token"] = tgt_token
            batch["tgt_mask"] = (tgt_token != self.pad_id).astype("int64")
            batch["tgt_pos"] = tgt_pos
            batch["tgt_type"] = tgt_role
            batch["tgt_turn"] = tgt_turn

        if "act" in samples[0]:
            act = [sp["act"] for sp in samples]
            batch["act_index"] = np.array(act)

        return batch, batch_size

class GalaxyRunner(object):
    def __init__(self, cfg, reader):
        self.cfg = cfg
        self.reader = reader
        self.optimizer = None
        self.lr_scheduler = None
        self.model = self.load_model()
        self.load()
        self.iterator = MultiWOZIterator(reader)
        self.evaluator = MultiWozEvaluator(reader, cfg.pred_data_type)

    def save(self, epoch, is_best=False):
        """ save """
        if not os.path.exists(self.cfg.save_dir):
            os.makedirs(self.cfg.save_dir)

        train_state = {"epoch": epoch,
                       "best_valid_metric": self.best_valid_metric,
                       "optimizer": self.optimizer.state_dict()}
        if self.lr_scheduler is not None:
            train_state["lr_scheduler"] = self.lr_scheduler.state_dict()

        # Save checkpoint
        # if self.save_checkpoint:
        #     model_file = os.path.join(self.save_dir, f"state_epoch_{self.epoch}.model")
        #     torch.save(self.model.state_dict(), model_file)
        #     self.logger.info(f"Saved model state to '{model_file}'")

        #     train_file = os.path.join(self.save_dir, f"state_epoch_{self.epoch}.train")
        #     torch.save(train_state, train_file)
        #     self.logger.info(f"Saved train state to '{train_file}'")

        # Save current best model
        if is_best:
            best_model_file = os.path.join(self.cfg.save_dir, "best.model")
            torch.save(self.model.state_dict(), best_model_file)
            best_train_file = os.path.join(self.cfg.save_dir, "best.train")
            torch.save(train_state, best_train_file)
            logger.info(
                f"Saved best model state to '{best_model_file}' with new best valid metric "
                f"combined score={self.best_valid_metric:.3f}")

    def load(self):
        """ load """
        def _load_model_state():
            model_state_dict = torch.load(f'{self.model.init_checkpoint}.model',
                                          map_location=lambda storage, loc: storage)

            if 'module.' in list(model_state_dict.keys())[0]:
                new_model_state_dict = OrderedDict()
                for k, v in model_state_dict.items():
                    assert k[:7] == 'module.'
                    new_model_state_dict[k[7:]] = v
                model_state_dict = new_model_state_dict

            new_model_state_dict = OrderedDict()
            parameters = {name: param for name, param in self.model.named_parameters()}
            for name, param in model_state_dict.items():
                if name in parameters:
                    if param.shape != parameters[name].shape:
                        assert hasattr(param, "numpy")
                        arr = param.numpy()
                        z = np.random.normal(scale=self.model.initializer_range,
                                             size=parameters[name].shape).astype("float32")
                        if name == 'embedder.token_embedding.weight':
                            z[-param.shape[0]:] = arr
                            print(f"part of parameter({name}) random normlize initialize")
                        else:
                            if z.shape[0] < param.shape[0]:
                                z = arr[:z.shape[0]]
                                print(f"part of parameter({name}) are dropped")
                            else:
                                z[:param.shape[0]] = arr
                                print(f"part of parameter({name}) random normlize initialize")
                        dtype, device = param.dtype, param.device
                        z = torch.tensor(z, dtype=dtype, device=device)
                        new_model_state_dict[name] = z
                    else:
                        new_model_state_dict[name] = param
                else:
                    print(f"parameter({name}) are dropped")
            model_state_dict = new_model_state_dict

            for name in parameters:
                if name not in model_state_dict:
                    if parameters[name].requires_grad:
                        print(f"parameter({name}) random normlize initialize")
                        z = np.random.normal(scale=self.model.initializer_range,
                                             size=parameters[name].shape).astype("float32")
                        dtype, device = parameters[name].dtype, parameters[name].device
                        model_state_dict[name] = torch.tensor(z, dtype=dtype, device=device)
                    else:
                        model_state_dict[name] = parameters[name]

            self.model.load_state_dict(model_state_dict)
            logger.info(f"Loaded model state from '{self.model.init_checkpoint}.model'")

        def _load_train_state():
            train_file = f"{self.model.init_checkpoint}.train"
            if os.path.exists(train_file):
                train_state_dict = torch.load(train_file, map_location=lambda storage, loc: storage)
                self.epoch = train_state_dict["epoch"]
                self.best_valid_metric = train_state_dict["best_valid_metric"]
                if self.optimizer is not None and "optimizer" in train_state_dict:
                    self.optimizer.load_state_dict(train_state_dict["optimizer"])
                if self.lr_scheduler is not None and "lr_scheduler" in train_state_dict:
                    self.lr_scheduler.load_state_dict(train_state_dict["lr_scheduler"])
                logger.info(
                    f"Loaded train state from '{train_file}' with (epoch-{self.epoch} "
                    f"best_valid_metric={self.best_valid_metric:.3f})")
            else:
                logger.info(f"Loaded no train state")

        if self.model.init_checkpoint is None:
            logger.info(f"Loaded no model !!!")
            return

        _load_model_state()
        _load_train_state()

    def load_model(self):
        self.cfg.Model.num_token_embeddings = self.reader.vocab_size
        self.cfg.Model.num_turn_embeddings = self.cfg.max_ctx_turn + 1

        generator = Generator.create(self.cfg, reader=self.reader)
        model = ModelBase.create(self.cfg, generator=generator)
        logger.info("Total number of parameters in networks is {}".format(sum(x.numel() for x in model.parameters())))
        
        model = model.to(self.cfg.device)

        return model

    def set_optimizers(self, num_training_steps):
        """
        Setup the optimizer and the learning rate scheduler.

        from transformers.Trainer

        parameters from cfg: lr (1e-3); warmup_steps
        """
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.cfg.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.cfg.lr)

        num_warmup_steps = self.cfg.warmup_steps if self.cfg.warmup_steps >= 0 else int(num_training_steps * 0.1)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        logger.info("Total training steps = {}, warmup steps = {}".format(num_training_steps, num_warmup_steps))

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def to_tensor(self, array):
        """
        numpy array -> tensor
        """
        array = torch.tensor(array)
        return array.to(self.cfg.device)

    def train(self):
        collate_fn = CollateForGalaxy(self.reader.pad_id, self.cfg.max_len, self.cfg.max_ctx_turn)
        train_dataset = MultiWOZDatasetGalaxy(self.reader.data['train'], 'train', self.reader.tokenizer)
        train_dataloader = DataLoader(train_dataset, batch_size=self.cfg.batch_size, shuffle=True, collate_fn=collate_fn)
        num_training_steps_per_epoch = len(train_dataloader) // self.cfg.gradient_accumulation_steps
        self.set_optimizers(num_training_steps=num_training_steps_per_epoch * self.cfg.epochs)

        self.best_valid_metric = 0.0
        best_epoch=0

        for epoch in range(1, self.cfg.epochs + 1):
            self.model.train()
            self.model.zero_grad()
            training_avg_loss = 0

            for step, turn_batch in enumerate(tqdm(train_dataloader, desc='Epoch {} Training'.format(epoch))):
                batch, batch_size = turn_batch
                batch = type(batch)(map(lambda kv: (kv[0], self.to_tensor(kv[1])), batch.items()))
                metrics = self.model(batch, is_training=True)
                loss = metrics['loss']


                if self.cfg.gradient_accumulation_steps > 1:
                    loss = loss / self.cfg.gradient_accumulation_steps
                
                training_avg_loss += loss.item()
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)

                if ((step + 1) % self.cfg.gradient_accumulation_steps == 0) or (step + 1 == len(train_dataloader)):
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
            
            logger.info("done {}/{} epoch; Average training loss: {}".format(epoch, self.cfg.epochs, training_avg_loss / len(train_dataloader)))

            if epoch > self.cfg.test_after_epochs:
            # if epoch > 0:
                bleu, success, match = self.predict(predict_when_training=True)
                score = 0.5 * (success + match) + bleu
                logger.info('Epoch %d: match: %2.2f; success: %2.2f; bleu: %2.2f; score: %.2f' % (
                    epoch, match, success, bleu, score))

                if score > self.best_valid_metric:
                    self.best_valid_metric = score
                    best_epoch = epoch
                    self.save(epoch, is_best=True)

                logger.info('Best combined score: {} at epoch {}.'.format(self.best_valid_metric, best_epoch))

    def predict(self, predict_when_training=False):
        self.model.eval()
        
        collate_fn = CollateForGalaxy(self.reader.pad_id, self.cfg.max_len, self.cfg.max_ctx_turn)
        pred_batches, _, _, _ = self.iterator.get_batches(self.cfg.pred_data_type, self.cfg.batch_size * 4, num_gpus=1)
        results = {}
        for dial_batch in tqdm(pred_batches, total=len(pred_batches), desc='Prediction'):
            batch_size = len(dial_batch)
            dial_history = [[] for _ in range(batch_size)]
            for turn_batch in self.iterator.transpose_batch(dial_batch):
                batch_bs_encoder_inputs_ids = []
                for t, turn in enumerate(turn_batch):
                    context = dial_history[t] + [turn['user']]
                    batch_bs_encoder_inputs_ids.append({'src': context})

                batch, _ = collate_fn(batch_bs_encoder_inputs_ids)
                batch = type(batch)(map(lambda kv: (kv[0], self.to_tensor(kv[1])), batch.items()))
                prompt_id = self.reader.sos_b_id
                with torch.no_grad():
                    outputs = self.model.infer(inputs=batch, start_id=prompt_id,
                                                    eos_id=self.reader.eos_b_id, max_gen_len=60)
                generated_bs = outputs.cpu().numpy().tolist()                    
                decoded_belief_outputs = self.finalize_outputs(generated_bs, 'bspn_gen', self.reader.eos_b_id)

                for t, turn in enumerate(turn_batch):
                    turn.update(**decoded_belief_outputs[t])

                for turn in turn_batch:
                    bspn_gen = turn['bspn_gen']
                    bspn_gen = self.reader.tokenizer.decode(bspn_gen, clean_up_tokenization_spaces=False)
                    db_token = self.reader.bspn_to_db_pointer(bspn_gen, turn['turn_domain'])
                    assert len(turn['dbpn']) == 4
                    booking_pointer = turn['dbpn'][2]

                    # yet to check
                    dbpn_gen = [self.reader.sos_d_id] + self.reader.tokenizer.convert_tokens_to_ids([db_token]) + [booking_pointer] + [self.reader.eos_d_id]
                    turn['dbpn_gen'] = dbpn_gen

                prev_inputs = []
                for t, turn in enumerate(turn_batch):
                    prev_inputs.append(turn['bspn_gen'] + turn['dbpn_gen'])
                prompt_id = self.reader.sos_a_id
                with torch.no_grad():
                    act_resp_outputs = self.model.infer(inputs=batch, start_id=prompt_id,
                                                        eos_id=self.reader.eos_r_id, max_gen_len=80,
                                                        prev_input=prev_inputs)
                act_resp_outputs = act_resp_outputs.cpu().numpy().tolist()
                decoded_act_resp_output = self.finalize_action_resp(act_resp_outputs)

                for t, turn in enumerate(turn_batch):
                    turn.update(**decoded_act_resp_output[t])
            
                # update dial_history
                for t, turn in enumerate(turn_batch):
                    dial_history[t].append(turn['user'])
                    dial_history[t].append(turn['bspn_gen'] + turn['dbpn_gen'] + turn['aspn_gen'] + turn['resp_gen'])
            
            result = self.iterator.get_readable_batch(dial_batch)
            results.update(**result)

        # 算分
        if predict_when_training == False:
            save_json(results, self.cfg.init_checkpoint + '_inference.json')
        
        bleu, success, match = self.evaluator.e2e_eval(results)
        return bleu, success, match
                
    def finalize_action_resp(self, resp_outputs):
        batch_decoded = []
        for resp_output in resp_outputs:
            try:
                bos_action_idx = resp_output.index(self.reader.sos_a_id)
                eos_action_idx = resp_output.index(self.reader.eos_a_id)
            except ValueError:
                aspn = [self.reader.sos_a_id, self.reader.eos_a_id]
            else:
                aspn = resp_output[bos_action_idx:eos_action_idx + 1]

            try:
                bos_resp_idx = resp_output.index(self.reader.sos_r_id)
                eos_resp_idx = resp_output.index(self.reader.eos_r_id)
            except ValueError:
                resp = [self.reader.sos_r_id, self.reader.eos_r_id]
            else:
                resp = resp_output[bos_resp_idx:eos_resp_idx + 1]

            decoded = {"aspn_gen": aspn, "resp_gen": resp}

            batch_decoded.append(decoded)

        return batch_decoded
            

    def finalize_outputs(self, outputs, output_type, eos_token_id):
        '''
        output_type: bspn_gen, aspn_gen, resp_gen
        '''
        batch_decoded = []
        for i, belief_output in enumerate(outputs):
            if eos_token_id not in belief_output:
                eos_idx = len(belief_output) - 1
            else:
                eos_idx = belief_output.index(eos_token_id)

            decoded = {}
            decoded[output_type] = belief_output[:eos_idx + 1]

            batch_decoded.append(decoded)

        return batch_decoded

class GalaxyReader(object):
    def __init__(self, cfg, version) -> None:
        self.version = version
        self.cfg = cfg
        self.nlp = spacy.load("en_core_web_sm")
        self.tokenizer = self.init_tokenizer()
        self.data_dir = self.get_data_dir()
        self.db = MultiWozDB(os.path.join(os.path.dirname(self.get_data_dir()), "db"))
        self.vocab_size = len(self.tokenizer)

        encoded_data_path = os.path.join(self.data_dir, "encoded_data_{}.pkl".format(cfg.model_name))

        if os.path.exists(encoded_data_path):
            logger.info("Load encoded data from {}".format(encoded_data_path))
            self.data = load_pickle(encoded_data_path)
        else:
            logger.info("Encode data and save to {}".format(encoded_data_path))
            train = self.encode_data("train")
            dev = self.encode_data("dev")
            test = self.encode_data("test")

            self.data = {"train": train, "dev": dev, "test": test}

            save_pickle(self.data, encoded_data_path)

        self.pad_token = '[PAD]'
        self.bos_token = '[BOS]'
        self.eos_token = '[EOS]'
        self.unk_token = '[UNK]'
        self.pad_id = self.tokenizer.convert_tokens_to_ids([self.pad_token])[0]
        self.bos_id = self.tokenizer.convert_tokens_to_ids([self.bos_token])[0]
        self.eos_id = self.tokenizer.convert_tokens_to_ids([self.eos_token])[0]
        self.unk_id = self.tokenizer.convert_tokens_to_ids([self.unk_token])[0]

        self.sos_b_id = self.tokenizer.convert_tokens_to_ids(['<sos_b>'])[0]
        self.eos_b_id = self.tokenizer.convert_tokens_to_ids(['<eos_b>'])[0]
        self.sos_a_id = self.tokenizer.convert_tokens_to_ids(['<sos_a>'])[0]
        self.eos_a_id = self.tokenizer.convert_tokens_to_ids(['<eos_a>'])[0]
        self.sos_d_id = self.tokenizer.convert_tokens_to_ids(['<sos_d>'])[0]
        self.eos_d_id = self.tokenizer.convert_tokens_to_ids(['<eos_d>'])[0]
        self.sos_r_id = self.tokenizer.convert_tokens_to_ids(['<sos_r>'])[0]
        self.eos_r_id = self.tokenizer.convert_tokens_to_ids(['<eos_r>'])[0]

    def bspn_to_db_pointer(self, bspn, turn_domain):
        constraint_dict = self.bspn_to_constraint_dict(bspn)

        matnums = self.db.get_match_num(constraint_dict)
        match_dom = turn_domain[0] if len(turn_domain) == 1 else turn_domain[1]
        match_dom = match_dom[1:-1] if match_dom.startswith("[") else match_dom
        match = matnums[match_dom]

        vector = self.db.addDBIndicator(match_dom, match)

        return vector

    def encode_data(self, data_type):
        data = load_json(os.path.join(self.data_dir, "{}_data_galaxy.json".format(data_type)))

        encoded_data = []
        for fn, dial in tqdm(data.items(), desc=data_type, total=len(data)):
            encoded_dial = []
            for idx, t in enumerate(dial['log']):
                enc = {}
                enc['dial_id'] = fn
                enc['turn_num'] = t['turn_num']
                enc['turn_domain'] = t['turn_domain'].split()
                enc["pointer"] = [int(i) for i in t["pointer"].split(",")]

                enc['user'] = [self.tokenizer.convert_tokens_to_ids(['<sos_u>'])[0]] + \
                    self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(t['user'])) + \
                    [self.tokenizer.convert_tokens_to_ids(['<eos_u>'])[0]]
                enc['resp'] = [self.tokenizer.convert_tokens_to_ids(['<sos_r>'])[0]] + \
                    self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(t['nodelx_resp'])) + \
                    [self.tokenizer.convert_tokens_to_ids(['<eos_r>'])[0]]
                enc['redx'] = [self.tokenizer.convert_tokens_to_ids(['<sos_r>'])[0]] + \
                    self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(t['resp'])) + \
                    [self.tokenizer.convert_tokens_to_ids(['<eos_r>'])[0]]
                
                constraint_dict = self.bspn_to_constraint_dict(t["constraint"])
                ordered_constraint_dict = OrderedDict()
                for domain, slots in definitions.INFORMABLE_SLOTS.items():
                    if domain not in constraint_dict:
                        continue

                    ordered_constraint_dict[domain] = OrderedDict()
                    for slot in slots:
                        if slot not in constraint_dict[domain]:
                            continue

                        value = constraint_dict[domain][slot]

                        ordered_constraint_dict[domain][slot] = value

                ordered_bspn = self.constraint_dict_to_bspn(ordered_constraint_dict)
                enc["bspn"] = [self.tokenizer.convert_tokens_to_ids(['<sos_b>'])[0]] + \
                    self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(ordered_bspn)) + \
                    [self.tokenizer.convert_tokens_to_ids(['<eos_b>'])[0]]
                enc['aspn'] = [self.tokenizer.convert_tokens_to_ids(['<sos_a>'])[0]] + \
                    self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(t['sys_act'])) + \
                    [self.tokenizer.convert_tokens_to_ids(['<eos_a>'])[0]]
                enc['user_aspn'] = [self.tokenizer.convert_tokens_to_ids(['<sos_a>'])[0]] + \
                    self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(t['user_act'])) + \
                    [self.tokenizer.convert_tokens_to_ids(['<eos_a>'])[0]]
                enc["goal_state"] = [self.tokenizer.convert_tokens_to_ids([definitions.BOS_GOAL_TOEKN])[0]] + \
                    self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(t['goal_state'])) + \
                    [self.tokenizer.convert_tokens_to_ids([definitions.EOS_GOAL_TOKEN])[0]]
                enc['user_aspn'] = [self.tokenizer.convert_tokens_to_ids(['<sos_a>'])[0]] + \
                    self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(t['user_act'])) + \
                    [self.tokenizer.convert_tokens_to_ids(['<eos_a>'])[0]]

                pointer = enc["pointer"][:-2]
                if not any(pointer):
                    db_token = definitions.DB_NULL_TOKEN
                else:
                    db_token = "[db_{}]".format(pointer.index(1))

                # 加入book标记
                if enc['pointer'][-2:] == [0, 1]:
                    book_pointer = '[book_success]'
                elif enc['pointer'][-2:] == [1, 0]:
                    book_pointer = '[book_fail]'
                else:
                    assert enc['pointer'][-2:] == [0, 0]
                    book_pointer = '[book_nores]'
                db_book_pointer = ' '.join([db_token, book_pointer])

                enc["dbpn"] = [self.tokenizer.convert_tokens_to_ids(['<sos_d>'])[0]] + \
                    self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(db_book_pointer)) + \
                    [self.tokenizer.convert_tokens_to_ids(['<eos_d>'])[0]]

                if 'unified_act' in t:
                    enc['act'] = [int(a) for a in t['unified_act'].split(',')]

                encoded_dial.append(enc)
            
            encoded_data.append(encoded_dial)
        
        return encoded_data

    def constraint_dict_to_bspn(self, constraint_dict):
        tokens = []
        for domain, sv_dict in constraint_dict.items():
            tokens.append("[" + domain + "]")
            for s, v in sv_dict.items():
                tokens.append("[value_" + s + "]")
                tokens.extend(v.split())

        return " ".join(tokens)

    def bspn_to_constraint_dict(self, bspn):
        bspn = bspn.split() if isinstance(bspn, str) else bspn

        eos_belief_token = '<eos_b>'

        constraint_dict = OrderedDict()
        domain, slot = None, None
        for token in bspn:
            if token == eos_belief_token:
                break

            if token.startswith("["):
                token = token[1:-1]

                if token in definitions.ALL_DOMAINS:
                    domain = token

                if token.startswith("value_"):
                    if domain is None:
                        continue

                    if domain not in constraint_dict:
                        constraint_dict[domain] = OrderedDict()

                    slot = token.split("_")[1]

                    constraint_dict[domain][slot] = []

            else:
                try:
                    if domain is not None and slot is not None:
                        constraint_dict[domain][slot].append(token)
                except KeyError:
                    continue

        for domain, sv_dict in constraint_dict.items():
            for s, value_tokens in sv_dict.items():
                constraint_dict[domain][s] = " ".join(value_tokens)

        return constraint_dict

    def get_all_special_tokens(self):
        special_tokens = ['[PAD]', '[BOS]', '[EOS]', '[UNK]']
        special_tokens.extend(['<sos_u>', '<eos_u>', '<sos_r>', '<eos_r>', '<sos_b>', '<eos_b>', '<sos_a>', '<eos_a>', '<sos_d>', '<eos_d>'])

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
        
        special_tokens.extend(definitions.DB_TOKENS)
        special_tokens.extend(definitions.DB_STATE_TOKENS)
        special_tokens.extend(definitions.GOAL_TOKENS)
        special_tokens.extend(definitions.USER_ACTION_TOEKNS)
        special_tokens.extend(['[book_success]', '[book_fail]', '[book_nores]'])

        return special_tokens

    def get_batches(self, data_type, batch_size, num_gpus, shuffle=False, num_dialogs=-1, excluded_domains=None):
        dial = self.reader.data[data_type]

        if excluded_domains is not None:
            logger.info("Exclude domains: {}".format(excluded_domains))

            target_dial_ids = []
            for domains, dial_ids in self.dial_by_domain.items():
                domain_list = domains.split("-")

                if len(set(domain_list) & set(excluded_domains)) == 0:
                    target_dial_ids.extend(dial_ids)

            dial = [d for d in dial if d[0]["dial_id"] in target_dial_ids]

        if num_dialogs > 0:
            dial = random.sample(dial, min(num_dialogs, len(dial)))

        turn_bucket = self.bucket_by_turn(dial)

        all_batches = []

        num_training_steps = 0
        num_turns = 0
        num_dials = 0
        for k in turn_bucket:
            if data_type != "test" and (k == 1 or k >= 17):
                continue

            batches = self.construct_mini_batch(
                turn_bucket[k], batch_size, num_gpus)

            num_training_steps += k * len(batches)
            num_turns += k * len(turn_bucket[k])
            num_dials += len(turn_bucket[k])
            all_batches += batches

        if shuffle:
            random.shuffle(all_batches)

        return all_batches, num_training_steps, num_dials, num_turns

    def init_tokenizer(self):
        special_tokens = self.get_all_special_tokens()
        tokenizer = Tokenizer(self.cfg.vocab_path, special_tokens=special_tokens, tokenizer_type='Bert')
        return tokenizer

    def get_data_dir(self):
        return os.path.join("data", "MultiWOZ_{}".format(self.version), "processed")

def parse_config():
    parser = argparse.ArgumentParser()
    # dataset configuration
    parser.add_argument("--version", type=str, default="2.0", choices=["2.0", "2.1"])

    # model configuration
    parser.add_argument("--model_name", type=str, default='galaxy', help = 'mttod, pptod, ubar, galaxy')
    parser.add_argument("--vocab_path", type=str, default='./galaxy_model/Bert/vocab.txt')
    parser.add_argument("--gpu", type=int, default=1)

    # training configuration
    parser.add_argument('--token_loss', type=bool, default=True)
    parser.add_argument('--run_type', type=str, default='train', choices=['train', 'predict'])
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--max_ctx_turn", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_to_keep_ckpt", type=int, default=1) 
    parser.add_argument("--save_dir", type=str, default='galaxy_model_finetune', help="directory to save the model parameters.")
    parser.add_argument("--pred_data_type", type=str, default='test', choices=['test', 'dev'])
    parser.add_argument("--output", type=str, default='inference.json', help="generated results")
    parser.add_argument("--test_after_epochs", type=int, default=5)
    parser.add_argument("--no_validation", action="store_true")
    parser.add_argument("--no_learning_rate_decay", action="store_true")

    # DDP
    parser.add_argument("--using_ddp", action="store_true")
    parser.add_argument("--gpu_num", type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    ModelBase.add_cmdline_argument(parser)
    Generator.add_cmdline_argument(parser)
    
    return parse_args(parser)

if __name__ == '__main__':

    '''
    CUDA_VISIBLE_DEVICES=3 python galaxy_finetune.py --batch_size 4 --gradient_accumulation_steps 8 --init_checkpoint ./galaxy_model/GALAXY
    '''

    if torch.cuda.is_available():
        logger.info('Cuda is available.')
    cuda_available = torch.cuda.is_available()
    multi_gpu_training = False
    cfg = parse_config()

    if cuda_available:
        if torch.cuda.device_count() > 1:
            multi_gpu_training = True
            logger.info('Using Multi-GPU training, number of GPU is {}'.format(torch.cuda.device_count()))
            torch.cuda.set_device(cfg.local_rank)
            device = torch.device('cuda', cfg.local_rank)
            torch.distributed.init_process_group(backend='nccl')
        else:
            logger.info('Using single GPU training.')
    else:
        pass

    device = torch.device('cuda')
    setattr(cfg, "device", device)
    if cuda_available:
        setattr(cfg, 'use_gpu', cuda_available)

    if cfg.seed > 0:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

        logger.info("Set random seed to %d", cfg.seed)

    galaxy_reader = GalaxyReader(cfg, cfg.version)
    galaxy_runner = GalaxyRunner(cfg, galaxy_reader)

    if cfg.run_type == 'train':
        galaxy_runner.train()
    elif cfg.run_type == 'predict':
        bleu, success, match = galaxy_runner.predict()
        score = 0.5 * (success + match) + bleu
        logger.info('match: %2.2f; success: %2.2f; bleu: %2.2f; score: %.2f' % (
                    match, success, bleu, score))

    
