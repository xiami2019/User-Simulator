"""
   MTTOD: reader.py

   implements MultiWoz Training/Validation Data Feeder for MTTOD.

   This code is partially referenced from thu-spmi's damd-multiwoz repository:
   (https://github.com/thu-spmi/damd-multiwoz/blob/master/reader.py)

   Copyright 2021 ETRI LIRS, Yohan Lee
   Copyright 2019 Yichi Zhang

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import os
import spacy
import random
import difflib
from tqdm import tqdm
from difflib import get_close_matches
from itertools import chain
from collections import OrderedDict

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import T5Tokenizer

from utils import definitions
from utils.utils import load_json, load_pickle, save_pickle, get_or_create_logger
from external_knowledges import MultiWozDB

logger = get_or_create_logger(__name__)


class BaseIterator(object):
    def __init__(self, reader):
        self.reader = reader

        self.dial_by_domain = load_json("data/MultiWOZ_2.1/dial_by_domain.json")

    def bucket_by_turn(self, encoded_data):
        turn_bucket = {}
        for dial in encoded_data:
            turn_len = len(dial)
            if turn_len not in turn_bucket:
                turn_bucket[turn_len] = []

            turn_bucket[turn_len].append(dial)

        return OrderedDict(sorted(turn_bucket.items(), key=lambda i: i[0]))

    def construct_mini_batch(self, data, batch_size, num_gpus):
        all_batches = []
        batch = []
        for dial in data:
            batch.append(dial)
            if len(batch) == batch_size:
                all_batches.append(batch)
                batch = []

        # if remainder > 1/2 batch_size, just put them in the previous batch, otherwise form a new batch
        if (len(batch) % num_gpus) != 0:
            batch = batch[:-(len(batch) % num_gpus)]
        if len(batch) > 0.5 * batch_size:
            all_batches.append(batch)
        elif len(all_batches):
            all_batches[-1].extend(batch)
        else:
            all_batches.append(batch)

        return all_batches

    def transpose_batch(self, dial_batch):
        turn_batch = []
        turn_num = len(dial_batch[0])
        for turn in range(turn_num):
            turn_l = []
            for dial in dial_batch:
                this_turn = dial[turn]
                turn_l.append(this_turn)
            turn_batch.append(turn_l)
        return turn_batch

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

    def flatten_dial_history(self, dial_history, len_postfix, context_size=-1):
        if context_size > 0:
            context_size -= 1

        if context_size == 0:
            windowed_context = []
        elif context_size > 0:
            windowed_context = dial_history[-context_size:]
        else:
            windowed_context = dial_history

        ctx_len = sum([len(c) for c in windowed_context])

        # consider eos_token
        spare_len = self.reader.max_seq_len - len_postfix - 1
        while ctx_len >= spare_len:
            ctx_len -= len(windowed_context[0])
            windowed_context.pop(0)

        context = list(chain(*windowed_context))

        return context

    def tensorize(self, ids):
        return torch.tensor(ids, dtype=torch.long)

    def get_data_iterator(self, all_batches, task, ururu, add_auxiliary_task=False, context_size=-1):
        raise NotImplementedError


class MultiWOZIterator(BaseIterator):
    def __init__(self, reader):
        super(MultiWOZIterator, self).__init__(reader)

    def get_readable_batch(self, dial_batch):
        dialogs = {}

        decoded_keys = ["user", "resp", "redx", "bspn", "aspn", "dbpn",
                        "bspn_gen", "bspn_gen_with_span",
                        "dbpn_gen", "aspn_gen", "resp_gen", "user_aspn", "goal_state"]
        for dial in dial_batch:
            dial_id = dial[0]["dial_id"]

            dialogs[dial_id] = []

            for turn in dial:
                readable_turn = {}

                for k, v in turn.items():
                    if k in ["dial_id", "resp_span", "user_span"]:
                        continue
                    elif k in decoded_keys:
                        v = self.reader.tokenizer.decode(
                            v, clean_up_tokenization_spaces=False)
                        '''
                        if k == "user":
                            print(k, v)
                        '''
                    elif k == "pointer":
                        turn_doamin = turn["turn_domain"][-1]
                        v = self.reader.db.pointerBack(v, turn_doamin)
                    if k == "user_span" or k == "resp_span":
                        speaker = k.split("_")[0]
                        v_dict = {}
                        for domain, ss_dict in v.items():
                            v_dict[domain] = {}
                            for s, span in ss_dict.items():
                                v_dict[domain][s] = self.reader.tokenizer.decode(
                                    turn[speaker][span[0]: span[1]])
                        v = v_dict

                    readable_turn[k] = v

                dialogs[dial_id].append(readable_turn)

        return dialogs

    def get_data_iterator(self, training_type):
        # pick particular data iterator
        if training_type == 'ds':
            return self.get_data_iterator_ds
        elif training_type == 'us':
            return self.get_data_iterator_us

    def get_data_iterator_us(self, all_batches, ururu, context_size=-1):
        # data iterator for trianing user simulator
        for dial_batch in all_batches:
            batch_encoder_input_ids = []
            batch_resp_label_ids = [] # user action labels and user utterance labels

            for dial in dial_batch:
                dial_encoder_input_ids = []
                dial_resp_label_ids = []
                
                dial_history = []
                for turn in dial:
                    context = self.flatten_dial_history(
                        dial_history, len(turn['goal_state']), context_size
                    )
                    encoder_input_ids = context + turn['goal_state'] + [self.reader.eos_token_id]
                    resp = turn['user_aspn'] + turn['user']
                    resp_label_ids = resp + [self.reader.eos_token_id]

                    dial_encoder_input_ids.append(encoder_input_ids)
                    dial_resp_label_ids.append(resp_label_ids)

                    if ururu:
                        turn_text = turn['user'] + turn['redx']
                    else:
                        turn_text = turn['user'] + turn['bspn'] + turn['dbpn'] + turn['aspn'] + turn['redx']

                    dial_history.append(turn_text)

                batch_encoder_input_ids.append(dial_encoder_input_ids)
                batch_resp_label_ids.append(dial_resp_label_ids)
            
            # turn first
            batch_encoder_input_ids = self.transpose_batch(batch_encoder_input_ids)
            batch_resp_label_ids = self.transpose_batch(batch_resp_label_ids)

            num_turns = len(batch_encoder_input_ids)

            tensor_encoder_input_ids = []
            tensor_resp_label_ids = []
            for t in range(num_turns):
                tensor_encoder_input_ids = [self.tensorize(b) for b in batch_encoder_input_ids[t]]
                tensor_resp_label_ids = [self.tensorize(b) for b in batch_resp_label_ids[t]]
                tensor_encoder_input_ids = pad_sequence(tensor_encoder_input_ids, batch_first=True, padding_value=self.reader.pad_token_id)
                tensor_resp_label_ids = pad_sequence(tensor_resp_label_ids, batch_first=True, padding_value=self.reader.pad_token_id)

                yield tensor_encoder_input_ids, tensor_resp_label_ids, None

    def get_data_iterator_ds(self, all_batches, ururu, context_size=-1):
        # data iterator for training dialogue system
        for dial_batch in all_batches:
            batch_encoder_input_ids = []
            batch_belief_label_ids = []
            batch_resp_label_ids = []

            for dial in dial_batch:
                dial_encoder_input_ids = []
                dial_belief_label_ids = []
                dial_resp_label_ids = []

                dial_history = []
                for turn in dial:
                    context = self.flatten_dial_history(
                        dial_history, len(turn["user"]), context_size)

                    encoder_input_ids = context + turn["user"] + [self.reader.eos_token_id]

                    bspn = turn["bspn"]

                    bspn_label = bspn

                    belief_label_ids = bspn_label + [self.reader.eos_token_id]
                    resp = turn['dbpn'] + turn["aspn"] + turn["redx"]
                    resp_label_ids = resp + [self.reader.eos_token_id]

                    dial_encoder_input_ids.append(encoder_input_ids)
                    dial_belief_label_ids.append(belief_label_ids)
                    dial_resp_label_ids.append(resp_label_ids)

                    if ururu:
                        turn_text = turn["user"] + turn["redx"]
                    else:
                        turn_text = turn["user"] + bspn + turn["dbpn"] + turn["aspn"] + turn["redx"]

                    dial_history.append(turn_text)

                batch_encoder_input_ids.append(dial_encoder_input_ids)
                batch_belief_label_ids.append(dial_belief_label_ids)
                batch_resp_label_ids.append(dial_resp_label_ids)

            # turn first
            batch_encoder_input_ids = self.transpose_batch(batch_encoder_input_ids)
            batch_belief_label_ids = self.transpose_batch(batch_belief_label_ids)
            batch_resp_label_ids = self.transpose_batch(batch_resp_label_ids)

            num_turns = len(batch_encoder_input_ids)

            tensor_encoder_input_ids = []
            tensor_belief_label_ids = []
            tensor_resp_label_ids = []
            for t in range(num_turns):
                tensor_encoder_input_ids = [
                    self.tensorize(b) for b in batch_encoder_input_ids[t]]
                tensor_belief_label_ids = [
                    self.tensorize(b) for b in batch_belief_label_ids[t]]
                tensor_resp_label_ids = [
                    self.tensorize(b) for b in batch_resp_label_ids[t]]

                tensor_encoder_input_ids = pad_sequence(tensor_encoder_input_ids,
                                                        batch_first=True,
                                                        padding_value=self.reader.pad_token_id)
                tensor_belief_label_ids = pad_sequence(tensor_belief_label_ids,
                                                       batch_first=True,
                                                       padding_value=self.reader.pad_token_id)

                tensor_resp_label_ids = pad_sequence(tensor_resp_label_ids,
                                                     batch_first=True,
                                                     padding_value=self.reader.pad_token_id)

                yield tensor_encoder_input_ids, tensor_resp_label_ids, tensor_belief_label_ids


class BaseReader(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.nlp = spacy.load("en_core_web_sm")

        self.tokenizer = self.init_tokenizer()

        self.data_dir = self.get_data_dir()

        # encoded_data_path = os.path.join(self.data_dir, "encoded_data.pkl")
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

    def get_data_dir(self):
        raise NotImplementedError

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

        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

        return tokenizer

    @property
    def pad_token(self):
        return self.tokenizer.pad_token

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    def eos_token(self):
        return self.tokenizer.eos_token

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def unk_token(self):
        return self.tokenizer.unk_token

    @property
    def max_seq_len(self):
        return self.tokenizer.model_max_length

    @property
    def vocab_size(self):
        return len(self.tokenizer)

    def get_token_id(self, token):
        return self.tokenizer.convert_tokens_to_ids(token)

    def encode_text(self, text, bos_token=None, eos_token=None):
        tokens = text.split() if isinstance(text, str) else text

        assert isinstance(tokens, list)

        if bos_token is not None:
            if isinstance(bos_token, str):
                bos_token = [bos_token]

            tokens = bos_token + tokens

        if eos_token is not None:
            if isinstance(eos_token, str):
                eos_token = [eos_token]

            tokens = tokens + eos_token

        encoded_text = self.tokenizer.encode(" ".join(tokens))

        # except eos token
        if encoded_text[-1] == self.eos_token_id:
            encoded_text = encoded_text[:-1]

        return encoded_text

    def encode_data(self, data_type):
        raise NotImplementedError


class MultiWOZReader(BaseReader):
    def __init__(self, cfg, version):
        self.version = version
        self.db = MultiWozDB(os.path.join(os.path.dirname(self.get_data_dir()), "db"))

        super(MultiWOZReader, self).__init__(cfg)

    def get_data_dir(self):
        return os.path.join(
            "data", "MultiWOZ_{}".format(self.version), "processed")

    def encode_data(self, data_type):
        data = load_json(os.path.join(self.data_dir, "{}_data.json".format(data_type)))

        encoded_data = []
        for fn, dial in tqdm(data.items(), desc=data_type, total=len(data)):
            encoded_dial = []

            if self.cfg.model_name == 'pptod' or self.cfg.model_name == 'ubar':
                bos_user_token = '<sos_u>'
                eos_user_token = '<eos_u>'

                bos_resp_token = '<sos_r>'
                eos_resp_token = '<eos_r>'

                bos_bspn_token = '<sos_b>'
                eos_bspn_token = '<eos_b>'

                bos_sys_aspn_token = '<sos_a>'
                eos_sys_aspn_token = '<eos_a>'

                bos_usr_aspn_token = definitions.BOS_USER_ACTION_TOKEN # not defined in PPTOD
                eos_usr_aspn_token = definitions.EOS_USER_ACTION_TOKEN # not defined in PPTOD

                bos_goal_token = definitions.BOS_GOAL_TOEKN # not defined in PPTOD
                eos_goal_token = definitions.EOS_GOAL_TOKEN # not defined in PPTOD

                bos_dbpn_token = '<sos_d>' 
                eos_dbpn_token = '<eos_d>'

            else: # MTTOD
                bos_user_token = definitions.BOS_USER_TOKEN
                eos_user_token = definitions.EOS_USER_TOKEN

                bos_resp_token = definitions.BOS_RESP_TOKEN
                eos_resp_token = definitions.EOS_RESP_TOKEN

                bos_bspn_token = definitions.BOS_BELIEF_TOKEN
                eos_bspn_token = definitions.EOS_BELIEF_TOKEN

                bos_sys_aspn_token = definitions.BOS_ACTION_TOKEN
                eos_sys_aspn_token = definitions.EOS_ACTION_TOKEN

                bos_usr_aspn_token = definitions.BOS_USER_ACTION_TOKEN
                eos_usr_aspn_token = definitions.EOS_USER_ACTION_TOKEN

                bos_goal_token = definitions.BOS_GOAL_TOEKN
                eos_goal_token = definitions.EOS_GOAL_TOKEN

                bos_dbpn_token = definitions.BOS_DB_TOKEN
                eos_dbpn_token = definitions.EOS_DB_TOKEN

            for idx, t in enumerate(dial["log"]):
                enc = {}
                enc["dial_id"] = fn
                enc["turn_num"] = t["turn_num"]
                enc["turn_domain"] = t["turn_domain"].split()
                enc["pointer"] = [int(i) for i in t["pointer"].split(",")]

                target_domain = enc["turn_domain"][0] if len(enc["turn_domain"]) == 1 else enc["turn_domain"][1]

                target_domain = target_domain[1:-1]

                user_ids = self.encode_text(t["user"],
                                            bos_token=bos_user_token,
                                            eos_token=eos_user_token)

                enc["user"] = user_ids

                usdx_ids = self.encode_text(t["user_delex"],
                                            bos_token=bos_user_token,
                                            eos_token=eos_user_token)

                resp_ids = self.encode_text(t["nodelx_resp"],
                                            bos_token=bos_resp_token,
                                            eos_token=eos_resp_token)

                enc["resp"] = resp_ids

                redx_ids = self.encode_text(t["resp"],
                                            bos_token=bos_resp_token,
                                            eos_token=eos_resp_token)

                enc["redx"] = redx_ids

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

                bspn_ids = self.encode_text(ordered_bspn,
                                            bos_token=bos_bspn_token,
                                            eos_token=eos_bspn_token)

                enc["bspn"] = bspn_ids

                aspn_ids = self.encode_text(t["sys_act"],
                                            bos_token=bos_sys_aspn_token,
                                            eos_token=eos_sys_aspn_token)

                enc["aspn"] = aspn_ids

                user_aspn_ids = self.encode_text(t['user_act'], bos_token=bos_usr_aspn_token, eos_token=eos_usr_aspn_token)
                enc["user_aspn"] = user_aspn_ids

                goal_ids = self.encode_text(t['goal_state'], bos_token=bos_goal_token, eos_token=eos_goal_token)
                enc["goal_state"] = goal_ids

                pointer = enc["pointer"][:-2]
                if not any(pointer):
                    db_token = definitions.DB_NULL_TOKEN
                else:
                    db_token = "[db_{}]".format(pointer.index(1))

                dbpn_ids = self.encode_text(db_token,
                                            bos_token=bos_dbpn_token,
                                            eos_token=eos_dbpn_token)

                enc["dbpn"] = dbpn_ids

                if (len(enc["user"]) == 0 or len(enc["resp"]) == 0 or
                        len(enc["redx"]) == 0 or len(enc["bspn"]) == 0 or
                        len(enc["aspn"]) == 0 or len(enc["dbpn"]) == 0):
                    raise ValueError(fn, idx)

                encoded_dial.append(enc)

            encoded_data.append(encoded_dial)

        return encoded_data

    def bspn_to_constraint_dict(self, bspn):
        bspn = bspn.split() if isinstance(bspn, str) else bspn

        if self.cfg.model_name == 'pptod' or self.cfg.model_name == 'ubar':
            eos_belief_token = '<eos_b>'
        else:
            eos_belief_token = definitions.EOS_BELIEF_TOKEN

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

    def constraint_dict_to_bspn(self, constraint_dict):
        tokens = []
        for domain, sv_dict in constraint_dict.items():
            tokens.append("[" + domain + "]")
            for s, v in sv_dict.items():
                tokens.append("[value_" + s + "]")
                tokens.extend(v.split())

        return " ".join(tokens)

    def get_span(self, domain, text, delex_text, constraint_dict):
        span_info = {}

        if domain not in constraint_dict:
            return span_info

        tokens = text.split() if isinstance(text, str) else text

        delex_tokens = delex_text.split() if isinstance(delex_text, str) else delex_text

        seq_matcher = difflib.SequenceMatcher()

        seq_matcher.set_seqs(tokens, delex_tokens)

        for opcode in seq_matcher.get_opcodes():
            tag, i1, i2, j1, j2 = opcode

            lex_tokens = tokens[i1: i2]
            delex_token = delex_tokens[j1: j2]

            if tag == "equal" or len(delex_token) != 1:
                continue

            delex_token = delex_token[0]

            if not delex_token.startswith("[value_"):
                continue

            slot = delex_token[1:-1].split("_")[1]

            if slot not in definitions.EXTRACTIVE_SLOT:
                continue

            value = self.tokenizer.convert_tokens_to_string(lex_tokens)

            if slot in constraint_dict[domain] and value in constraint_dict[domain][slot]:
                if domain not in span_info:
                    span_info[domain] = {}

                span_info[domain][slot] = (i1, i2)

        return span_info

    def bspn_to_db_pointer(self, bspn, turn_domain):
        constraint_dict = self.bspn_to_constraint_dict(bspn)

        matnums = self.db.get_match_num(constraint_dict)
        match_dom = turn_domain[0] if len(turn_domain) == 1 else turn_domain[1]
        match_dom = match_dom[1:-1] if match_dom.startswith("[") else match_dom
        match = matnums[match_dom]

        vector = self.db.addDBIndicator(match_dom, match)

        return vector

    def canonicalize_span_value(self, domain, slot, value, cutoff=0.6):
        ontology = self.db.extractive_ontology

        if domain not in ontology or slot not in ontology[domain]:
            return value

        candidates = ontology[domain][slot]

        matches = get_close_matches(value, candidates, n=1, cutoff=cutoff)

        if len(matches) == 0:
            return value
        else:
            return matches[0]
