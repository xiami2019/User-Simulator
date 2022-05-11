'''
2022.4.14
你说人生艳丽我没有异议
你说人生忧郁我不言语
'''
import os
import json
import torch
import random
from tqdm import tqdm
from transformers import GPT2Tokenizer, BertTokenizer
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from utils.utils import definitions, get_or_create_logger, load_pickle, save_pickle, load_json
from itertools import chain

logger = get_or_create_logger(__name__)

class Lm_Reader(object):
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.tokenizer = self.init_tokenizer()
        self.data_dir = os.path.join("data", "MultiWOZ_{}".format(self.cfg.version), "processed")
    
        if self.cfg.text_file is None:
            if not self.cfg.compute_for_single:
                if self.cfg.ppl_level == 'sentence' or self.cfg.ppl_level == 'bart_score':
                    if self.cfg.gpt_score_singe_side:
                        assert self.cfg.agent is not None
                        encoded_data_path = os.path.join(self.data_dir, "encoded_data_lm_sentence_{}.pkl".format(self.cfg.agent)) 
                    else:
                        encoded_data_path = os.path.join(self.data_dir, "encoded_data_lm_sentence.pkl") 

                # elif self.cfg.ppl_level == 'session':
                #     encoded_data_path = os.path.join(self.data_dir, "encoded_data_lm_session.pkl")   

                if os.path.exists(encoded_data_path):
                    logger.info("Load encoded data from {}".format(encoded_data_path))
                    self.data = load_pickle(encoded_data_path)
                else:
                    logger.info("Encode data and save to {}".format(encoded_data_path))
                    train = self.encode_data('train', self.cfg.ppl_level)
                    dev = self.encode_data('dev', self.cfg.ppl_level)
                    test = self.encode_data('test', self.cfg.ppl_level)

                    self.data = {'train': train, 'dev': dev, 'test': test}
                    save_pickle(self.data, encoded_data_path)
        else:
            logger.info("Encode data of {}".format(self.cfg.text_file))
            test = self.encode_data_for_text_file()
            self.data = {'test': test}
    
    def encode_data_for_text_file(self):
        text_data = load_json(self.cfg.text_file)
        encoded_data = []

        if self.cfg.ppl_level == 'sentence' or self.cfg.ppl_level == 'bart_score':
            for dial in tqdm(text_data):
                for turn in dial['log']:
                    if self.cfg.gpt_score_singe_side:
                        assert self.cfg.agent is not None
                        if self.cfg.agent == 'usr':
                            user_idx = self.tokenizer.encode(turn['user']) + [self.tokenizer.eos_token_id]
                            if len(user_idx) > 1:
                                encoded_data.append(user_idx)
                        elif self.cfg.agent == 'sys':
                            resp_idx = self.tokenizer.encode(turn['sys']) + [self.tokenizer.eos_token_id]
                            if len(resp_idx) > 1:
                                encoded_data.append(resp_idx)
                    else:
                        user_idx = self.tokenizer.encode(turn['user']) + [self.tokenizer.eos_token_id]
                        resp_idx = self.tokenizer.encode(turn['sys']) + [self.tokenizer.eos_token_id]
                        if len(user_idx) > 1:
                            encoded_data.append(user_idx)
                        if len(resp_idx) > 1:
                            encoded_data.append(resp_idx)
        # elif self.cfg.ppl_level == 'session':
        #     bos_user_id = self.tokenizer.convert_tokens_to_ids(definitions.BOS_USER_TOKEN)
        #     eos_user_id = self.tokenizer.convert_tokens_to_ids(definitions.EOS_USER_TOKEN)
        #     bos_resp_id = self.tokenizer.convert_tokens_to_ids(definitions.BOS_RESP_TOKEN)
        #     eos_resp_id = self.tokenizer.convert_tokens_to_ids(definitions.EOS_RESP_TOKEN)

        #     for dial in tqdm(text_data):
        #         single_dial_ids = []
        #         for turn in dial['log']:
        #             user_idx = self.tokenizer.encode(turn['user'])
        #             resp_idx = self.tokenizer.encode(turn['sys'])
        #             single_dial_ids += [bos_user_id] + user_idx + [eos_user_id]
        #             single_dial_ids += [bos_resp_id] + resp_idx + [eos_resp_id]
        #         encoded_data.append(single_dial_ids + [self.tokenizer.eos_token_id])

        return encoded_data

    def init_tokenizer(self):
        if self.cfg.ckpt is not None:
            logger.info('Load tokenizer from {}'.format(self.cfg.ckpt))
            return GPT2Tokenizer.from_pretrained(self.cfg.ckpt)
        else:
            tokenizer = GPT2Tokenizer.from_pretrained(self.cfg.backbone)
        
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

        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

        return tokenizer

    def encode_data(self, data_type, data_level):
        data = load_json(os.path.join(self.data_dir, "{}_data.json".format(data_type)))

        encoded_data = []
        max_len = 0
        if data_level == 'sentence' or data_level == 'bart_score':
            for fn, dial in tqdm(data.items(), desc=data_type, total=len(data)):
                for idx, t in enumerate(dial['log']):
                    if self.cfg.gpt_score_singe_side:
                        assert self.cfg.agent is not None
                        if self.cfg.agent == 'usr':
                            user_idx = self.tokenizer.encode(t['user']) + [self.tokenizer.eos_token_id]
                            encoded_data.append(user_idx)
                        elif self.cfg.agent == 'sys':
                            resp_idx = self.tokenizer.encode(t['resp']) + [self.tokenizer.eos_token_id]
                            encoded_data.append(resp_idx)
                    else:
                        user_idx = self.tokenizer.encode(t['user']) + [self.tokenizer.eos_token_id]
                        resp_idx = self.tokenizer.encode(t['resp']) + [self.tokenizer.eos_token_id]
                        encoded_data.append(user_idx)
                        encoded_data.append(resp_idx)
        # elif data_level == 'session':
        #     bos_user_id = self.tokenizer.convert_tokens_to_ids(definitions.BOS_USER_TOKEN)
        #     eos_user_id = self.tokenizer.convert_tokens_to_ids(definitions.EOS_USER_TOKEN)
        #     bos_resp_id = self.tokenizer.convert_tokens_to_ids(definitions.BOS_RESP_TOKEN)
        #     eos_resp_id = self.tokenizer.convert_tokens_to_ids(definitions.EOS_RESP_TOKEN)

        #     max_len = 0
        #     for fn, dial in tqdm(data.items(), desc=data_type, total=len(data)):
        #         single_dial_ids = []
        #         for idx, t in enumerate(dial['log']):
        #             user_idx = self.tokenizer.encode(t['user'])
        #             resp_idx = self.tokenizer.encode(t['resp'])
        #             single_dial_ids += [bos_user_id] + user_idx + [eos_user_id]
        #             single_dial_ids += [bos_resp_id] + resp_idx + [eos_resp_id]
        #         encoded_data.append(single_dial_ids + [self.tokenizer.eos_token_id])
        #         max_len = max(max_len, len(single_dial_ids))
        #     logger.info('Max Len is {}'.format(max_len))

        return encoded_data

class Bert_Reader(object):
    def __init__(self, cfg, ) -> None:
        self.cfg = cfg
        self.tokenizer = self.init_tokenizer()
        self.data_dir = os.path.join("data", "MultiWOZ_{}".format(self.cfg.version), "processed")

        if self.cfg.text_file is None:
            encoded_data_path = os.path.join(self.data_dir, "encoded_data_nsp_1.pkl")
            if os.path.exists(encoded_data_path):
                logger.info("Load encoded data from {}".format(encoded_data_path))
                self.data = load_pickle(encoded_data_path)
            else:
                logger.info("Encode data and save to {}".format(encoded_data_path))
                train = self.encode_data('train')
                dev = self.encode_data('dev')
                test = self.encode_data('test')

                self.data = {'train': train, 'dev': dev, 'test': test}
                save_pickle(self.data, encoded_data_path)
        else:
            logger.info("Encode data of {}".format(self.cfg.text_file))
            test = self.encode_data_for_text_file()
            self.data = {'test': test}

    def init_tokenizer(self):
        if self.cfg.ckpt is not None:
            logger.info('Load tokenizer from {}'.format(self.cfg.ckpt))
            return BertTokenizer.from_pretrained(self.cfg.ckpt)
        else:
            logger.info('Load tokenizer from {}'.format(self.cfg.backbone))
            tokenizer = BertTokenizer.from_pretrained(self.cfg.backbone)

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

        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

        return tokenizer

    def encode_data_for_text_file(self):
        text_data = load_json(self.cfg.text_file)
        encoded_data = []

        for dial in tqdm(text_data):
            single_dial = []
            for turn in dial['log']:
                user_ids = self.tokenizer.encode(turn['user'])[1:-1] # 去掉BERT的CLS和SEP
                resp_ids = self.tokenizer.encode(turn['sys'])[1:-1]
                single_dial.append(user_ids)
                single_dial.append(resp_ids)
            encoded_data.append(single_dial)

        return encoded_data

    def encode_data(self, data_type):
        data = load_json(os.path.join(self.data_dir, "{}_data.json".format(data_type)))

        encoded_data = []
        for fn, dial in tqdm(data.items(), desc=data_type):
            single_dial = []
            for idx, t in enumerate(dial['log']):
                user_ids = self.tokenizer.encode(t['user'])[1:-1] # 去掉BERT的CLS和SEP
                resp_ids = self.tokenizer.encode(t['resp'])[1:-1]
                single_dial.append(user_ids)
                single_dial.append(resp_ids)
            encoded_data.append(single_dial)
        
        return encoded_data               

class Collate_Fn(object):
    def __init__(self, pad_token_id) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        batch_tensor = [torch.tensor(i, dtype=torch.long) for i in batch]
        batch_tensor = pad_sequence(batch_tensor, batch_first=True, padding_value=self.pad_token_id)
        return batch_tensor

class Collate_Fn_NSP(object):
    def __init__(self, pad_token_id) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        batch_input_ids = []
        batch_label_ids = []
        for i in batch:
            batch_input_ids.append(i[0])
            batch_label_ids.append(i[1])

        batch_label_tensor = torch.tensor(batch_label_ids, dtype=torch.long)
        batch_input_tensor = [torch.tensor(i, dtype=torch.long) for i in batch_input_ids]
        batch_input_tensor = pad_sequence(batch_input_tensor, batch_first=True, padding_value=self.pad_token_id)
        return batch_input_tensor, batch_label_tensor

class MultiwozDataset(Dataset):
    def __init__(self, tokenizer, data, type):
        super().__init__()
        self.tokenizer = tokenizer
        self.data = data
        self.type = type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class MultiwozNSPDataset(Dataset):
    def __init__(self, tokenizer, data, type) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.type = type
        self.data = self.construct_nsp_data(data)

    def construct_nsp_data(self, data):
        nsp_data = []

        if self.type == 'test':
            for j, dial in enumerate(data):
                for i in range(1, len(dial)):
                    nsp_data.append({'data': [self.tokenizer.cls_token_id] + dial[i-1] + [self.tokenizer.sep_token_id] + dial[i], 'label': 0})
        else:
            for j, dial in enumerate(data):
                for i in range(1, len(dial)):
                    # positive examples
                    nsp_data.append({'data': [self.tokenizer.cls_token_id] + dial[i-1] + [self.tokenizer.sep_token_id] + dial[i], 'label': 0})
                    # negative examples
                    prob = random.random()
                    # 正负例 1:1
                    if prob < 0.5:
                        # 50%概率是本session中的随机语句；
                        while True:
                            neg_sen_idx = random.randint(0, len(dial) - 1)
                            if neg_sen_idx != i:
                                break
                        nsp_data.append({'data': [self.tokenizer.cls_token_id] + dial[i-1] + [self.tokenizer.sep_token_id] + dial[neg_sen_idx], 'label': 1})
                    else:
                        # 50%概率是别的session中的随机语句；
                        while True:
                            neg_dial_idx = random.randint(0, len(data) - 1)
                            if neg_dial_idx != j:
                                break

                        neg_sen_idx = random.randint(0, len(data[neg_dial_idx]) - 1)
                        nsp_data.append({'data': [self.tokenizer.cls_token_id] + dial[i-1] + [self.tokenizer.sep_token_id] + data[neg_dial_idx][neg_sen_idx], 'label': 1})
        
        return nsp_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]['data'], self.data[idx]['label']

class MultiwozBartScoreDataset(Dataset):
    '''
    do not tokenize, so do not need reader;
    '''
    def __init__(self, text_file) -> None:
        super().__init__()
        self.data = self.construct_bartscore_data(text_file)

    def construct_bartscore_data(self, text_file):
        bart_score_data = []
        if text_file == 'test' or text_file == 'dev':
            with open('./data/MultiWOZ_2.0/processed/bart_score_{}_data.json'.format(text_file), 'r') as f:
                for line in f.readlines():
                    turn = json.loads(line)
                    bart_score_data.append((turn['text'], turn['summary']))
        else:
            text_data = load_json(text_file)
            for dial in text_data:
                history = 'session starts.'
                for turn in dial['log']:
                    bart_score_data.append((history, turn['user']))
                    history += ' ' + turn['user']
                    bart_score_data.append((history, turn['sys']))
                    history += ' ' + turn['sys']

        return bart_score_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1]

