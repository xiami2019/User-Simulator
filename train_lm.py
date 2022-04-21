'''
2022.4.14
我可以很久不和你联络
任日子一天这么过

2022.4.15
不要你离开
距离隔不开
思念变成海
在窗外进不来
'''
import os
import glob
import math
import torch
import random
import shutil
import argparse
import numpy as np
import torch.nn as nn
from abc import *
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from transformers import GPT2LMHeadModel, BertForNextSentencePrediction, AdamW, get_linear_schedule_with_warmup, get_constant_schedule
from lm_dataset import Lm_Reader, Bert_Reader, MultiwozDataset, MultiwozNSPDataset, Collate_Fn, Collate_Fn_NSP
from torch.utils.data import DataLoader
from utils.utils import get_or_create_logger

logger = get_or_create_logger(__name__)

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-backbone', type=str, default='gpt2', choices=['gpt2', 'bert-base-uncased'])
    parser.add_argument('-ckpt', type=str, default=None)
    parser.add_argument('-version', type=str, default='2.0', choices=['2.0', '2.1'])
    parser.add_argument('-seed', type=int, default=42)
    parser.add_argument('-run_type', type=str, default='train', choices=['train', 'predict'])
    parser.add_argument('-max_to_keep_ckpt', type=int, default=1)
    parser.add_argument('-epochs', type=int, default=20)
    parser.add_argument('-grad_accum_steps', type=int, default=1)
    parser.add_argument('-warmup_steps', type=int, default=-1)
    parser.add_argument('-warmup_ratio', type=float, default=0.2)
    parser.add_argument('-learning_rate', type=float, default=1e-4)
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-max_grad_norm', type=float, default=1.0)
    parser.add_argument('-log_frequency', type=int, default=100)
    parser.add_argument('-model_dir', type=str, default='gpt_lm_model')
    parser.add_argument('-no_learning_rate_decay', action="store_true")
    parser.add_argument('-text_file', type=str, default=None)
    parser.add_argument('-ppl_level', type=str, default='session', choices=['sentence', 'session'])
    parser.add_argument('-early_stopping', type=int, default=5)
    parser.add_argument('-task', type=str, choices=['ppl', 'nsp'])
    parser.add_argument('-nsp_score', type=str, default='hard', choices=['soft', 'hard'])

    return parser.parse_args()

class BaseRunner(metaclass=ABCMeta):
    def __init__(self, cfg, reader) -> None:
        self.cfg = cfg
        self.reader = reader
        self.model = self.load_model()
    
    def load_model(self):
        if self.cfg.ckpt is not None:
            model_path = self.cfg.ckpt
        else:
            model_path = self.cfg.backbone
        
        if self.cfg.backbone in ['bert-base-uncased']:
            model = BertForNextSentencePrediction.from_pretrained(model_path)
        elif self.cfg.backbone in ['gpt2']:
            model = GPT2LMHeadModel.from_pretrained(model_path)

        logger.info('Load model from {}'.format(model_path))

        model.resize_token_embeddings(len(self.reader.tokenizer))
        model.to(self.cfg.device)
        return model

    def get_optimizer_and_scheduler(self, num_training_steps_per_epoch):
        num_train_steps = (num_training_steps_per_epoch *
            self.cfg.epochs) // self.cfg.grad_accum_steps

        if self.cfg.warmup_steps >= 0:
            num_warmup_steps = self.cfg.warmup_steps
        else:
            num_warmup_steps = int(num_train_steps * self.cfg.warmup_ratio)

        logger.info("Total training steps = {}, warmup steps = {}".format(
            num_train_steps, num_warmup_steps))

        optimizer = AdamW(self.model.parameters(), lr=self.cfg.learning_rate)

        if self.cfg.no_learning_rate_decay:
            scheduler = get_constant_schedule(optimizer)
        else:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_train_steps)

        return optimizer, scheduler

    def save_model(self, epoch):
        latest_ckpt = "ckpt-epoch{}".format(epoch)
        save_path = os.path.join(self.cfg.model_dir, latest_ckpt)

        model = self.model
        model.save_pretrained(save_path)
        self.reader.tokenizer.save_pretrained(save_path)

        # keep chekpoint up to maximum
        checkpoints = sorted(
            glob.glob(os.path.join(self.cfg.model_dir, "ckpt-*")),
            key=os.path.getmtime,
            reverse=True)

        checkpoints_to_be_deleted = checkpoints[self.cfg.max_to_keep_ckpt:]

        for ckpt in checkpoints_to_be_deleted:
            shutil.rmtree(ckpt)

        return latest_ckpt
    
    @abstractclassmethod
    def train(self):
        raise NotImplementedError

    @abstractclassmethod
    def validation(self, type):
        raise NotImplementedError

class BertRunner(BaseRunner):
    def __init__(self, cfg, reader):
        super().__init__(cfg, reader)

    def train(self):
        train_dataset = MultiwozNSPDataset(self.reader.tokenizer, self.reader.data['train'], 'train')
        collate_fn = Collate_Fn_NSP(self.reader.tokenizer.pad_token_id)
        train_dataLoader = DataLoader(dataset=train_dataset, shuffle=True, collate_fn=collate_fn, num_workers=4, batch_size=self.cfg.batch_size)
        num_training_steps_per_epoch = len(train_dataLoader)
        optimizer, scheduler = self.get_optimizer_and_scheduler(num_training_steps_per_epoch)
        best_acc = 0
        best_epoch = 0
        stop_count = 0

        for epoch in range(1, self.cfg.epochs + 1):
            self.model.train()
            self.model.zero_grad()
            training_avg_loss = 0

            for step, batch in enumerate(tqdm(train_dataLoader, desc='Epoch {} Traning'.format(epoch))):
                input_ids, label_ids = batch
                input_ids = input_ids.to(self.cfg.device)
                label_ids = label_ids.to(self.cfg.device)
                attention_mask = torch.where(input_ids == self.reader.tokenizer.pad_token_id, 0, 1)

                model_outputs = self.model(
                    input_ids = input_ids,
                    attention_mask=attention_mask,
                    labels=label_ids,
                )

                loss = model_outputs.loss
                training_avg_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if self.cfg.log_frequency > 0 and (step + 1) % self.cfg.log_frequency == 0:
                    tqdm.write('Epoch: {}; Batch: {}; Loss: {}'.format(epoch, step + 1, loss.item()))

            current_acc = self.validation('dev')
            if current_acc > best_acc:
                stop_count = 0
                best_acc = current_acc
                best_epoch = epoch
                self.save_model(epoch)
            else:
                stop_count += 1

            logger.info('Done {}/{} epoch: avg training loss: {:.6};'.format(epoch, self.cfg.epochs, training_avg_loss / num_training_steps_per_epoch))
            logger.info('Current validation Acc: {:.3}; Best Acc is {:.3} at epoch {};'.format(current_acc, best_acc, best_epoch))            

            if stop_count >= self.cfg.early_stopping:
                logger.info('Early stopped. Stop count is {}'.format(self.cfg.early_stopping))
                break               

    def validation(self, type):
        self.model.eval()
        valid_dataset = MultiwozNSPDataset(self.reader.tokenizer, self.reader.data[type], type)
        collate_fn = Collate_Fn_NSP(self.reader.tokenizer.pad_token_id)
        valid_dataLodaer = DataLoader(dataset=valid_dataset, shuffle=False, collate_fn=collate_fn, num_workers=4, batch_size=self.cfg.batch_size)
        total_correct = 0
        total_examples = 0
        total_soft_score = 0

        for _, batch in enumerate(tqdm(valid_dataLodaer, desc='Validation')):
            input_ids, label_ids = batch
            input_ids = input_ids.to(self.cfg.device)
            label_ids = label_ids.to(self.cfg.device)
            attention_mask = torch.where(input_ids == self.reader.tokenizer.pad_token_id, 0, 1)

            with torch.no_grad():
                model_outputs = self.model(
                    input_ids = input_ids,
                    attention_mask = attention_mask,
                    label_ids = label_ids,
                )
            softmax = nn.Softmax(dim=1)
            logits = model_outputs.logits
            logits = softmax(logits)
            pred = torch.argmax(logits, dim=1)
            soft_score = logits[:, 0].sum()
            total_soft_score += soft_score
            correct = torch.where(pred == label_ids, 1, 0).sum()
            total_correct += correct
            total_examples += len(pred)

        acc = total_correct / total_examples
        score = total_soft_score / total_examples

        if self.cfg.nsp_score == 'hard':
            return acc
        elif self.cfg.nsp_score == 'soft':
            return score

class LMRunner(BaseRunner):
    def __init__(self, cfg, reader) -> None:
        super().__init__(cfg, reader)

    def train(self):
        train_dataset = MultiwozDataset(self.reader.tokenizer, self.reader.data['train'], 'train')
        collate_fn = Collate_Fn(self.reader.tokenizer.eos_token_id)
        train_dataLodaer = DataLoader(dataset=train_dataset, shuffle=True, collate_fn=collate_fn, num_workers=4, batch_size=self.cfg.batch_size)
        num_training_steps_per_epoch = len(train_dataLodaer)
        optimizer, scheduler = self.get_optimizer_and_scheduler(num_training_steps_per_epoch)
        best_ppl = float('inf')
        best_epoch = 0
        stop_count = 0

        for epoch in range(1, self.cfg.epochs + 1):
            self.model.train()
            self.model.zero_grad()
            training_avg_loss = 0

            for step, input_ids in enumerate(tqdm(train_dataLodaer, desc='Epoch {} Traning'.format(epoch))):
                input_ids = input_ids.to(self.cfg.device)
                attention_mask = torch.where(input_ids == self.reader.tokenizer.eos_token_id, 0, 1)
                
                model_outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids,
                )
                loss = model_outputs.loss
                training_avg_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if self.cfg.log_frequency > 0 and (step + 1) % self.cfg.log_frequency == 0:
                    tqdm.write('Epoch: {}; Batch: {}; Loss: {:.8}'.format(epoch, step + 1, loss.item()))

            current_ppl, eval_loss = self.validation('dev')
            if current_ppl < best_ppl:
                stop_count = 0
                best_ppl = current_ppl
                best_epoch = epoch
                self.save_model(epoch)
            else:
                stop_count += 1

            logger.info('Done {}/{} epoch: avg training loss: {:.6}; validation loss:{:.6}'.format(epoch, self.cfg.epochs, training_avg_loss / num_training_steps_per_epoch, eval_loss))
            logger.info('Current validation PPL: {:.3}; Best PPL is {:.3} at epoch {};'.format(current_ppl, best_ppl, best_epoch))            

            if stop_count >= self.cfg.early_stopping:
                logger.info('Early stopped. Stop count is {}'.format(self.cfg.early_stopping))
                break

    def validation(self, type):
        self.model.eval()

        valid_dataset = MultiwozDataset(self.reader.tokenizer, self.reader.data[type], type)
        collate_fn = Collate_Fn(self.reader.tokenizer.eos_token_id)
        valid_dataLodaer = DataLoader(dataset=valid_dataset, shuffle=False, collate_fn=collate_fn, num_workers=4, batch_size=self.cfg.batch_size)
        total_token = 0
        eval_loss = 0
        nlls = []

        for _, input_ids in enumerate(tqdm(valid_dataLodaer, desc='Validation')):
            input_ids = input_ids.to(self.cfg.device)
            attention_mask = torch.where(input_ids == self.reader.tokenizer.eos_token_id, 0, 1)
            with torch.no_grad():
                model_outputs = self.model(
                    input_ids=input_ids,
                    attention_mask = attention_mask,
                    labels=input_ids,
                )
            eval_loss += model_outputs.loss.item()
            
            target_len = attention_mask.sum(dim=1)
            logits = model_outputs.logits
            logits_without_padding = [logits[i][:int(target_len[i])+1][:-1] for i in range(logits.shape[0])]
            labels_without_padding = [input_ids[i][:int(target_len[i])+1][1:] for i in range(input_ids.shape[0])]

            loss_fct = CrossEntropyLoss(reduction='mean')

            for i in range(logits.shape[0]):
                loss = loss_fct(logits_without_padding[i], labels_without_padding[i])
                neg_log_likelihood = loss.item() * (target_len[i] + 1)
                nlls.append(neg_log_likelihood)
                total_token += target_len[i] + 1

        ppl = math.exp(sum(nlls) / total_token)
        return ppl, eval_loss / len(valid_dataLodaer)

def main():
    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setattr(cfg, 'device', device)

    # set random seed
    if cfg.seed > 0:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

        logger.info("Set random seed to %d", cfg.seed)

    if cfg.task == 'ppl':
        reader = Lm_Reader(cfg)
        runner = LMRunner(cfg, reader)
        if cfg.run_type == 'train':
            runner.train()
        elif cfg.run_type == 'predict':
            ppl, _ = runner.validation('test')
            logger.info("Test set PPL: {}".format(ppl))
    elif cfg.task == 'nsp':
        reader = Bert_Reader(cfg)
        runner = BertRunner(cfg, reader)
        if cfg.run_type == 'train':
            runner.train()
        elif cfg.run_type == 'predict':
            acc = runner.validation('test')
            logger.info("Test set Acc: {}".format(acc))


if __name__ == '__main__':
    main()


