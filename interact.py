import os
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from itertools import chain
from collections import OrderedDict
from types import SimpleNamespace
from torch.nn.utils.rnn import pad_sequence
from transformers import T5ForConditionalGeneration, T5Tokenizer, GPT2LMHeadModel, GPT2Tokenizer, BertForNextSentencePrediction, BertTokenizer
from utils.utils import load_json, save_json, convert_goal_dict_to_span, convert_generate_action_span_to_dict, \
update_goal_states_during_gen, get_or_create_logger, split_user_act_and_resp
from utils import definitions
from external_knowledges import MultiWozDB
from evaluator import MultiWozEvaluator, convert_results_format
from reader import MultiWOZReader

logger = get_or_create_logger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device2 = torch.device('cpu')
# if torch.cuda.device_count() == 2:
#     device2 = torch.device("cuda", 1)

'''
岁月难得沉默
秋风厌倦漂泊
夕阳赖着不走挂在墙头舍不得我
'''

# special tokens map
mttod_to_pptod = {
    '<bos_user>': '<sos_u>',
    '<eos_user>': '<eos_u>',
    '<bos_resp>': '<sos_r>',
    '<eos_resp>': '<eos_r>',
    '<bos_belief>': '<sos_b>',
    '<eos_belief>': '<eos_b>',
    '<bos_act>': '<sos_a>',
    '<eos_act>': '<eos_a>',
    '<bos_db>': '<sos_d>',
    '<eos_db>': '<eos_d>',
}
pptod_to_mttod = {
    '<sos_u>': '<bos_user>',
    '<eos_u>': '<eos_user>',
    '<sos_r>': '<bos_resp>',
    '<eos_r>': '<eos_resp>',
    '<sos_b>': '<bos_belief>',
    '<eos_b>': '<eos_belief>',
    '<sos_a>': '<bos_act>',
    '<eos_a>': '<eos_act>',
    '<sos_d>': '<bos_db>',
    '<eos_d>': '<eos_db>',
}

def get_config():
    parser = argparse.ArgumentParser(description='RL config')
    parser.add_argument("-rl_dial_one_epoch", type=int, default=200)
    parser.add_argument("-rl_batch_size", type=int, default=1)
    parser.add_argument("-epochs", type=int, default=20)
    parser.add_argument("-simulator_path", type=str, default='./simulator_t5_small/ckpt-epoch12')
    parser.add_argument("-dialog_sys_path", type=str, default='./dialogue_t5_small/ckpt-epoch11')
    parser.add_argument("-simulator_save_path", type=str, default=None)
    parser.add_argument("-dialog_save_path", type=str, default=None)
    # parser.add_argument("-simulator_path", type=str, default='./simulator_t5_small/simulator_rl_v5_epoch_7')
    # parser.add_argument("-dialog_sys_path", type=str, default='./dialogue_t5_small/dialog_rl_v5_epoch_7')
    parser.add_argument("-max_turn_num", type=int, default=20)
    parser.add_argument("-data_dir", type=str, default='./data/MultiWOZ_2.0/')
    parser.add_argument("-model_dir", type=str, default="simulator_t5_small")
    parser.add_argument("-discount_factor", type=float, default=0.99)
    parser.add_argument('-rl_lr', type=float, default=0.0001, help='learning rate for reinforcement learning')
    parser.add_argument('-grad_clip', type=float, default=1)
    parser.add_argument("-seed", type=int, default=1998)
    parser.add_argument('-do_rl_training', action="store_true")
    parser.add_argument('-use_ppl_as_reward', action="store_true")
    parser.add_argument('-ppl_ckpt', type=str, default='./gpt_lm_model_lr_1e_4_sentence/ckpt-epoch6')
    parser.add_argument('-use_nsp_score_as_reward', action="store_true")
    parser.add_argument('-nsp_ckpt', type=str, default='./bert_nsp_model_lr_1e_5_1/ckpt-epoch9')
    parser.add_argument('-gpt_score_ckpt', type=str, default='./bart_score_gpt_lm_model_lr_1e_4/ckpt-epoch6')
    parser.add_argument('-nsp_coef', type=float, default=0.5)
    parser.add_argument('-ppl_coef', type=float, default=0.5)
    parser.add_argument('-use_bart_score', action="store_true")
    parser.add_argument('-use_gpt_score_as_reward', action="store_true")
    parser.add_argument('-gpt_score_coef', type=float, default=0.5)
    parser.add_argument('-use_mean_rl_loss', action="store_true")
    parser.add_argument('-generate_results_path', type=str, default='generate_results.json')
    parser.add_argument('-model_name', type=str, default='mttod', choices=['mttod', 'ubar', 'pptod', 'galaxy'])
    args = parser.parse_args()

    return args

class InteractionEnvironment(object):
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.simulator_model = self.load_simulator(self.cfg.simulator_path)
        self.dialog_model = self.load_system(self.cfg.dialog_sys_path)
        self.simulator_tokenizer = self.load_simulator_tokenizer(self.cfg.simulator_path)
        self.dialog_tokenizer = self.load_sys_tokenizer(self.cfg.dialog_sys_path)
        self.data_dir = self.cfg.data_dir
        db_path = os.path.join(os.path.dirname(self.data_dir), 'db')
        logger.info("Load Database from {}".format(db_path))
        self.db = MultiWozDB(db_path)
        self.get_goal_list()

        # pptod prefix
        if self.cfg.model_name == 'pptod':
            bs_prefix_text = 'translate dialogue to belief state:'
            da_prefix_text = 'translate dialogue to dialogue action:'
            nlg_prefix_text = 'translate dialogue to system response:'
            self.bs_prefix_id = self.dialog_tokenizer.convert_tokens_to_ids(self.dialog_tokenizer.tokenize(bs_prefix_text))
            self.da_prefix_id = self.dialog_tokenizer.convert_tokens_to_ids(self.dialog_tokenizer.tokenize(da_prefix_text))
            self.nlg_prefix_id = self.dialog_tokenizer.convert_tokens_to_ids(self.dialog_tokenizer.tokenize(nlg_prefix_text))
            self.sos_context_token_id = self.dialog_tokenizer.convert_tokens_to_ids(['<sos_context>'])[0]
            self.eos_context_token_id = self.dialog_tokenizer.convert_tokens_to_ids(['<eos_context>'])[0]


    @property
    def all_goals(self):
        return self.goal_list

    def load_simulator(self, model_path):
        logger.info("Load simulator model from {}".format(model_path))
        if not os.path.exists(model_path):
            raise Exception('Model path is invalid!')
        return T5ForConditionalGeneration.from_pretrained(model_path)

    def load_system(self, model_path):
        if self.cfg.model_name == 'mttod':
            logger.info("Load system model from {}".format(model_path))
            if not os.path.exists(model_path):
                raise Exception('Model path is invalid!')
            return T5ForConditionalGeneration.from_pretrained(model_path)
        elif self.cfg.model_name == 'ubar':
            logger.info("Load system model from {}".format(model_path))
            if not os.path.exists(model_path):
                raise Exception('Model path is invalid!')
            return GPT2LMHeadModel.from_pretrained(model_path)
        elif self.cfg.model_name == 'pptod':
            logger.info("Load system model from {}".format(model_path))
            if not os.path.exists(model_path):
                raise Exception('Model path is invalid!')
            return T5ForConditionalGeneration.from_pretrained(model_path)
        elif self.cfg.model_name == 'galaxy':
            raise NotImplementedError

    def load_simulator_tokenizer(self, tokenizer_path):
        logger.info("Load simulator tokenizer from {}".format(tokenizer_path))
        if not os.path.exists(tokenizer_path):
            raise Exception('Tokenizer path is invalid!')
        return T5Tokenizer.from_pretrained(tokenizer_path)

    def load_sys_tokenizer(self, tokenizer_path):
        if self.cfg.model_name == 'mttod':
            logger.info("Load system tokenizer from {}".format(tokenizer_path))
            if not os.path.exists(tokenizer_path):
                raise Exception('Tokenizer path is invalid!')
            return T5Tokenizer.from_pretrained(tokenizer_path)
        elif self.cfg.model_name == 'ubar':
            logger.info("Load system tokenizer from {}".format(tokenizer_path))
            if not os.path.exists(tokenizer_path):
                raise Exception('Tokenizer path is invalid!')
            return GPT2Tokenizer.from_pretrained(tokenizer_path)
        elif self.cfg.model_name == 'pptod':
            logger.info("Load system tokenizer from {}".format(tokenizer_path))
            if not os.path.exists(tokenizer_path):
                raise Exception('Tokenizer path is invalid!')
            return T5Tokenizer.from_pretrained(tokenizer_path)
        elif self.cfg.model_name == 'galaxy':
            raise NotImplementedError
        
        logger.info("Load tokenizer from {}".format(tokenizer_path))
        if not os.path.exists(tokenizer_path):
            raise Exception('Tokenizer path is invalid!')
        return T5Tokenizer.from_pretrained(tokenizer_path)
        
    def get_goal_list(self):
        train_data_path = os.path.join(self.data_dir, 'processed', 'train_data.json')
        valid_data_path = os.path.join(self.data_dir, 'processed', 'dev_data.json')
        test_data_path = os.path.join(self.data_dir, 'processed', 'test_data.json')
        train_data = load_json(train_data_path)
        valid_data = load_json(valid_data_path)
        test_data = load_json(test_data_path)
        data = {'train': train_data, 'valid': valid_data, 'test': test_data}
        self.goal_list = {'train': [], 'valid': [], 'test': []}
        for data_type in data:
            for dialog_id, session in data[data_type].items():
                self.goal_list[data_type].append({'dialog_id': dialog_id, 'goal': session['goal']})
        assert len(data['train']) == len(self.goal_list['train'])
        assert len(data['valid']) == len(self.goal_list['valid'])
        assert len(data['test']) == len(self.goal_list['test'])

    def flatten_dial_history(self, dial_history, len_postifx, max_length):
        ctx_len = sum([len(c) for c in dial_history])
        
        #consider eos_token
        spare_len = max_length - len_postifx - 1
        while ctx_len >= spare_len:
            ctx_len -= len(dial_history[0])
            dial_history.pop(0)

        context = list(chain(*dial_history))
        return context

    def tensorize(self, ids):
        return torch.tensor(ids, dtype=torch.long)

    def encode_text(self, text, tokenizer, bos_token=None, eos_token=None, special_tokens_map=None):
        tokens = text.split() if isinstance(text, str) else text
        assert isinstance(tokens, list)

        # replace special tokens
        if special_tokens_map != None:
            for i in range(len(tokens)):
                if tokens[i] in special_tokens_map:
                    tokens[i] = special_tokens_map[tokens[i]]
        
        if bos_token is not None:
            if isinstance(bos_token, str):
                bos_token = [bos_token]
            tokens = bos_token + tokens
        
        if eos_token is not None:
            if isinstance(eos_token, str):
                eos_token = [eos_token]
            tokens = tokens + eos_token

        encoded_text = tokenizer.encode(" ".join(tokens))
        # except eos token
        if encoded_text[-1] == tokenizer.eos_token_id:
            encoded_text = encoded_text[:-1]

        return encoded_text

    def split_system_act_and_resp(self, model_output, model_output_prob=None):
        if model_output_prob is not None:
            pad_tensor = torch.tensor([0,0,0,0]).to(device)
            model_output_prob = torch.cat((pad_tensor, model_output_prob)) # pad for db tokens

        bos_act_token_id = self.dialog_tokenizer.convert_tokens_to_ids(definitions.BOS_ACTION_TOKEN)
        eos_act_token_id = self.dialog_tokenizer.convert_tokens_to_ids(definitions.EOS_ACTION_TOKEN)
        
        bos_resp_token_id = self.dialog_tokenizer.convert_tokens_to_ids(definitions.BOS_RESP_TOKEN)
        eos_resp_token_id = self.dialog_tokenizer.convert_tokens_to_ids(definitions.EOS_RESP_TOKEN)

        eos_token_id = self.dialog_tokenizer.eos_token_id

        if eos_token_id in model_output:
            eos_idx = model_output.index(eos_token_id)
            model_output = model_output[:eos_idx]
            if model_output_prob is not None:
                model_output_prob = model_output_prob[:eos_idx]

        # aspn
        aspn_prob = None
        if bos_act_token_id in model_output and eos_act_token_id in model_output:
            bos_action_idx = model_output.index(bos_act_token_id)
            eos_action_idx = model_output.index(eos_act_token_id)
            aspn = model_output[bos_action_idx:eos_action_idx + 1]
            if model_output_prob is not None:
                aspn_prob = model_output_prob[bos_action_idx:eos_action_idx + 1]
        else:
            aspn = [bos_act_token_id, eos_act_token_id]
            if model_output_prob is not None:
                aspn_prob = model_output_prob[:2]
                aspn_prob = 0 * aspn_prob
        
        resp_prob = None
        if bos_resp_token_id in model_output and eos_resp_token_id in model_output:
            bos_resp_token_idx = model_output.index(bos_resp_token_id)
            eos_resp_token_idx = model_output.index(eos_resp_token_id)
            resp = model_output[bos_resp_token_idx:eos_resp_token_idx+1]
            if model_output_prob is not None:
                resp_prob = model_output_prob[bos_resp_token_idx:eos_resp_token_idx+1]
        elif eos_act_token_id in model_output:
            eos_action_idx = len(model_output) - model_output[::-1].index(eos_act_token_id) - 1
            resp = model_output[eos_action_idx + 1:]
            if model_output_prob is not None:
                resp_prob = model_output_prob[eos_action_idx + 1:]

            if resp[-1] != eos_resp_token_id:
                resp.append(eos_resp_token_id)
                if model_output_prob is not None:
                    pad_tensor = torch.tensor([1.0]).to(device)
                    resp_prob = torch.cat((resp_prob, pad_tensor))
            if resp[0] != bos_resp_token_id:
                resp = [bos_resp_token_id] + resp
                if model_output_prob is not None:
                    pad_tensor = torch.tensor([1.0]).to(device)
                    resp_prob = torch.cat((pad_tensor, resp_prob))
        else:
            resp = [bos_resp_token_id, eos_resp_token_id]
            if model_output_prob is not None:
                resp_prob = model_output_prob[:2]
                resp_prob = 0 * resp_prob

        return aspn, resp, aspn_prob, resp_prob

    def finalize_bspn(self, belief_outputs, belief_states_prob=None):
        if self.cfg.model_name == 'mttod':
            eos_belief_token_id = self.dialog_tokenizer.convert_tokens_to_ids(definitions.EOS_BELIEF_TOKEN)
        else:
            eos_belief_token_id = self.dialog_tokenizer.convert_tokens_to_ids('<eos_b>')
        if belief_outputs[0] == self.dialog_tokenizer.pad_token_id:
            belief_outputs = belief_outputs[1:]
        if belief_outputs[-1] == self.dialog_tokenizer.eos_token_id:
            belief_outputs = belief_outputs[:-1]
            if belief_states_prob is not None:
                belief_states_prob = belief_states_prob[:-1]
        if eos_belief_token_id not in belief_outputs:
            eos_idx = len(belief_outputs) - 1
        else:
            eos_idx = belief_outputs.index(eos_belief_token_id)

        if belief_states_prob is not None:
            return belief_outputs[:eos_idx+1], belief_states_prob[:eos_idx+1]
        else:
            return belief_outputs[:eos_idx+1], None

    def finalize_aspn(self, aspn_outputs, aspn_states_prob=None):
        if self.cfg.model_name == 'mttod':
            eos_action_token_id = self.dialog_tokenizer.convert_tokens_to_ids(definitions.EOS_ACTION_TOKEN)
        else:
            eos_action_token_id = self.dialog_tokenizer.convert_tokens_to_ids('<eos_a>')
        if aspn_outputs[0] == self.dialog_tokenizer.pad_token_id:
            aspn_outputs = aspn_outputs[1:]
        if aspn_outputs[-1] == self.dialog_tokenizer.eos_token_id:
            aspn_outputs = aspn_outputs[:-1]
            if aspn_states_prob is not None:
                aspn_states_prob = aspn_states_prob[:-1]
        if eos_action_token_id not in aspn_outputs:
            eos_idx = len(aspn_outputs) - 1
        else:
            eos_idx = aspn_outputs.index(eos_action_token_id)

        if aspn_states_prob is not None:
            return aspn_outputs[:eos_idx+1], aspn_states_prob[:eos_idx+1]
        else:
            return aspn_outputs[:eos_idx+1], None

    def finalize_resp(self, resp_outputs, resp_states_prob=None):
        if self.cfg.model_name == 'mttod':
            eos_resp_token_id = self.dialog_tokenizer.convert_tokens_to_ids(definitions.EOS_RESP_TOKEN)
        else:
            eos_resp_token_id = self.dialog_tokenizer.convert_tokens_to_ids('<eos_r>')
        if resp_outputs[0] == self.dialog_tokenizer.pad_token_id:
            resp_outputs = resp_outputs[1:]
        if resp_outputs[-1] == self.dialog_tokenizer.eos_token_id:
            resp_outputs = resp_outputs[:-1]
            if resp_states_prob is not None:
                resp_states_prob = resp_states_prob[:-1]
        if eos_resp_token_id not in resp_outputs:
            eos_idx = len(resp_outputs) - 1
        else:
            eos_idx = resp_outputs.index(eos_resp_token_id)

        if resp_states_prob is not None:
            return resp_outputs[:eos_idx+1], resp_states_prob[:eos_idx+1]
        else:
            return resp_outputs[:eos_idx+1], None

    def bspn_to_constraint_dict(self, bspn):
        bspn = bspn.split() if isinstance(bspn, str) else bspn

        if self.cfg.model_name == 'mttod':
            eos_belief_token = definitions.EOS_BELIEF_TOKEN
        else:
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

    def constraint_dict_to_bspn(self, constraint_dict):
        tokens = []
        for domain, sv_dict in constraint_dict.items():
            tokens.append("[" + domain + "]")
            for s, v in sv_dict.items():
                tokens.append("[value_" + s + "]")
                tokens.extend(v.split())

        return " ".join(tokens)

    def bspn_to_db_pointer(self, bspn, turn_domain):
        constraint_dict = self.bspn_to_constraint_dict(bspn)

        matnums = self.db.get_match_num(constraint_dict)
        match_dom = turn_domain[0] if len(turn_domain) == 1 else turn_domain[1]
        match_dom = match_dom[1:-1] if match_dom.startswith("[") else match_dom
        match = matnums[match_dom]

        db_token = self.db.addDBIndicator(match_dom, match)

        return db_token

    def generate_single_dialog(self, user_goal, with_logprob=False, agent=None):
        self.simulator_model.to(device)
        self.dialog_model.to(device)

        # clear fail info and invalid/prev_invalid field
        for domain in user_goal['goal']:
            if 'fail_info' in user_goal['goal'][domain]:
                del user_goal['goal'][domain]['fail_info']
            if 'fail_book' in user_goal['goal'][domain]:
                del user_goal['goal'][domain]['fail_book']

            if 'book' in user_goal['goal'][domain]:
                if 'invalid' in user_goal['goal'][domain]['book']:
                    del user_goal['goal'][domain]['book']['invalid']
                if 'pre_invalid' in user_goal['goal'][domain]['book']:
                    del user_goal['goal'][domain]['book']['pre_invalid']

        dial_gen = {user_goal['dialog_id']: {'goal': user_goal['goal']}}
        log = []
        dialog_history = []
        goal_state_dict = user_goal['goal']
        goal_state_span = convert_goal_dict_to_span(user_goal['goal'])
        user_utterance = None
        turn_domain = None
        system_act = None
        user_act = None
        utterance_count = 0
        single_turn = {}

        if with_logprob:
            output_scores=True
            return_dict_in_generate=True
        else:
            output_scores=False
            return_dict_in_generate=False

        if_sys_need_grad = True if agent is not None and agent == 'sys' else False
        if_usr_need_grad = True if agent is not None and agent == 'usr' else False

        def is_continue(dial_gen):
            if 'sys' not in single_turn and 'user' in single_turn:
                # end up with system resp
                return True
            if len(goal_state_dict) == 0:
                # goal清空后终止
                dial_gen['terminate_reason'] = 'goal清空后终止'
                return False
            if len(log) >= self.cfg.max_turn_num:
                # 超过固定轮数终止
                dial_gen['terminate_reason'] = '超过{}轮终止'.format(self.cfg.max_turn_num)
                return False
            if system_act and ('[bye]' in system_act or '[thank]' in system_act):
                dial_gen['terminate_reason'] = 'system said thank or bye'
                return False
            if user_act and ('[bye]' in user_act or '[thank]' in user_act):
                dial_gen['terminate_reason'] = 'user said thank or bye'
                return False
            # 不满足退出条件则继续循环
            return True

        while is_continue(dial_gen): # 需要判断一个会话是否结束，满足结束条件则需要退出循环
            if utterance_count & 1 :
                '''
                system agent:
                input: user + dialog history;
                output1: belief states;
                output2: action + response;
                update user's goal state;
                '''
                utterance_count += 1

                if user_utterance is None:
                    raise Exception('Should generate user utterance first!')
                
                user_utterance_ids = self.encode_text(user_utterance, self.dialog_tokenizer)
                encoded_dialog_history = [self.encode_text(text, self.dialog_tokenizer) for text in dialog_history]
                context = self.flatten_dial_history(encoded_dialog_history, len(user_utterance_ids), self.dialog_tokenizer.model_max_length)
                input_ids = self.tensorize([context + user_utterance_ids + [self.dialog_tokenizer.eos_token_id]])
                input_ids = input_ids.to(device)

                dialog_generate = self.dialog_model.generate.__wrapped__
                bspn_decoder_input_ids = self.tensorize([[self.dialog_tokenizer.pad_token_id] + [self.dialog_tokenizer.convert_tokens_to_ids(definitions.BOS_BELIEF_TOKEN)]])
                bspn_decoder_input_ids = bspn_decoder_input_ids.to(device)
                # belief states generation
                torch.set_grad_enabled(if_sys_need_grad)
                model_output = dialog_generate(
                    self.dialog_model,
                    input_ids=input_ids,
                    decoder_input_ids=bspn_decoder_input_ids,
                    eos_token_id=self.dialog_tokenizer.eos_token_id,
                    # max_length=100,
                    max_length=80,
                    output_scores=output_scores,
                    return_dict_in_generate=return_dict_in_generate,
                )
                torch.set_grad_enabled(True)
                if with_logprob:
                    belief_states_output = model_output.sequences.cpu().numpy().tolist()
                    belief_states_prob = torch.max(torch.stack(model_output.scores, dim=1).softmax(-1), dim=-1).values[0]
                else:
                    belief_states_output = model_output.cpu().numpy().tolist()
                    belief_states_prob = None
                bspn_gen, _ = self.finalize_bspn(belief_states_output[0])
                
                if with_logprob:
                    # assert len(bspn_gen) == len(belief_states_prob)
                    # single_turn['bspn_prob'] = belief_states_prob
                    single_turn['bspn_prob'] = belief_states_prob
                
                bspn_gen = self.dialog_tokenizer.decode(bspn_gen, clean_up_tokenization_spaces=False)
                single_turn['belief_states'] = bspn_gen
                
                
                if turn_domain is None:
                    raise Exception('Domain is empty')
                db_token = self.bspn_to_db_pointer(bspn_gen, turn_domain)
                dbpn_gen = self.encode_text(db_token, self.dialog_tokenizer, bos_token=definitions.BOS_DB_TOKEN, eos_token=definitions.EOS_DB_TOKEN)
                single_turn['dbpn'] = self.dialog_tokenizer.decode(dbpn_gen)
                dbpn_gen = [self.dialog_tokenizer.pad_token_id] + dbpn_gen

                resp_decoder_input_ids = self.tensorize([dbpn_gen])
                resp_decoder_input_ids = resp_decoder_input_ids.to(device)
                # response generation
                torch.set_grad_enabled(if_sys_need_grad)
                model_output = dialog_generate(
                    self.dialog_model,
                    input_ids=input_ids,
                    decoder_input_ids=resp_decoder_input_ids,
                    eos_token_id=self.dialog_tokenizer.eos_token_id,
                    # max_length=200,
                    max_length=100,
                    output_scores=output_scores,
                    return_dict_in_generate=return_dict_in_generate,
                )
                torch.set_grad_enabled(True)
                if with_logprob:
                    resp_outputs = model_output.sequences.cpu().numpy().tolist()
                    resp_prob = torch.max(torch.stack(model_output.scores, dim=1).softmax(-1), dim=-1).values[0]
                else:
                    resp_outputs = model_output.cpu().numpy().tolist()
                    resp_prob = None
                system_act, system_resp, sys_act_prob, sys_resp_prob = self.split_system_act_and_resp(resp_outputs[0])
                
                if with_logprob:
                    single_turn['sys_act_resp_prob'] = resp_prob
                
                system_act = self.dialog_tokenizer.decode(system_act, clean_up_tokenization_spaces=False).split()
                system_resp = self.dialog_tokenizer.decode(system_resp, clean_up_tokenization_spaces=False).split()
                system_act_dict = convert_generate_action_span_to_dict(system_act[1:-1])
                goal_state_dict = update_goal_states_during_gen(goal_state_dict, system_act_dict, 'sys')
                
                single_turn['sys_act'] = ' '.join(system_act[1:-1])
                single_turn['sys'] = ' '.join(system_resp[1:-1])

                # update dialog history
                dialog_history.append(user_utterance)
                dialog_history.append(system_resp)

                log.append(single_turn.copy())
                single_turn = {}

                user_utterance = None
                turn_domain = None

            else:
                '''
                user agent:
                input: dialog history + goal state span;
                output: user action + user utterance;
                update user's goal state;
                '''
                utterance_count += 1

                goal_state_span = convert_goal_dict_to_span(goal_state_dict)
                goal_state_ids = self.encode_text(goal_state_span, self.simulator_tokenizer, bos_token=definitions.BOS_GOAL_TOEKN, eos_token=definitions.EOS_GOAL_TOKEN)
                encoded_dialog_history = [self.encode_text(text, self.simulator_tokenizer) for text in dialog_history]
                context = self.flatten_dial_history(encoded_dialog_history, len(goal_state_ids), self.simulator_tokenizer.model_max_length)
                input_ids = self.tensorize([context + goal_state_ids + [self.simulator_tokenizer.eos_token_id]])
                input_ids = input_ids.to(device)

                generate_with_graph = self.simulator_model.generate.__wrapped__
                torch.set_grad_enabled(if_usr_need_grad)
                model_output = generate_with_graph(
                    self.simulator_model,
                    input_ids=input_ids,
                    eos_token_id=self.simulator_tokenizer.eos_token_id,
                    # max_length=200,
                    max_length=100,
                    output_scores=output_scores,
                    return_dict_in_generate=return_dict_in_generate,
                )
                torch.set_grad_enabled(True)
                if with_logprob:
                    user_utterance_output = model_output.sequences.cpu().numpy().tolist()
                    user_utterance_prob = torch.max(torch.stack(model_output.scores, dim=1).softmax(-1), dim=-1).values[0]
                else:
                    user_utterance_output = model_output.cpu().numpy().tolist()
                    user_utterance_prob = None

                user_act, user_utterance, user_act_prob, _ = split_user_act_and_resp(self.simulator_tokenizer, user_utterance_output[0])
                
                if with_logprob:
                    single_turn['user_act_resp_prob'] = user_utterance_prob


                user_act = self.simulator_tokenizer.decode(user_act, clean_up_tokenization_spaces=False).split(' ')
                user_utterance = self.simulator_tokenizer.decode(user_utterance, clean_up_tokenization_spaces=False).split(' ')

                if len(user_act[1:-1]) == 0 or user_act[1][1:-1] == 'general':
                    turn_domain = ['[general]']
                elif user_act[1][1:-1] not in definitions.ALL_DOMAINS:
                    # raise Exception('Invalid domain token')
                    turn_domain = ['[general]']
                else:
                    turn_domain = [user_act[1]]

                # only add user utterance to history
                single_turn['user'] = ' '.join(user_utterance[1:-1])
                single_turn['user_act'] = ' '.join(user_act[1:-1])

                # update goal state
                user_act_dict = convert_generate_action_span_to_dict(user_act[1:-1])
                goal_state_dict = update_goal_states_during_gen(goal_state_dict, user_act_dict, 'user')

        dial_gen['log'] = log
        dial_gen['final_goal_state'] = convert_goal_dict_to_span(goal_state_dict)
        return dial_gen
    
    def generate_single_dialog_pptod(self, user_goal):
        self.simulator_model.to(device)
        self.dialog_model.to(device)

        # clear fail info and invalid/prev_invalid field
        for domain in user_goal['goal']:
            if 'fail_info' in user_goal['goal'][domain]:
                del user_goal['goal'][domain]['fail_info']
            if 'fail_book' in user_goal['goal'][domain]:
                del user_goal['goal'][domain]['fail_book']

            if 'book' in user_goal['goal'][domain]:
                if 'invalid' in user_goal['goal'][domain]['book']:
                    del user_goal['goal'][domain]['book']['invalid']
                if 'pre_invalid' in user_goal['goal'][domain]['book']:
                    del user_goal['goal'][domain]['book']['pre_invalid']

        dial_gen = {user_goal['dialog_id']: {'goal': user_goal['goal']}}
        log = []
        dialog_history = []
        goal_state_dict = user_goal['goal']
        goal_state_span = convert_goal_dict_to_span(user_goal['goal'])
        user_utterance = None
        turn_domain = None
        system_act = None
        user_act = None
        utterance_count = 0
        single_turn = {}

        def is_continue(dial_gen):
            if 'sys' not in single_turn and 'user' in single_turn:
                # end up with system resp
                return True
            if len(goal_state_dict) == 0:
                # goal清空后终止
                dial_gen['terminate_reason'] = 'goal清空后终止'
                return False
            if len(log) >= self.cfg.max_turn_num:
                # 超过固定轮数终止
                dial_gen['terminate_reason'] = '超过{}轮终止'.format(self.cfg.max_turn_num)
                return False
            if system_act and ('[bye]' in system_act or '[thank]' in system_act):
                dial_gen['terminate_reason'] = 'system said thank or bye'
                return False
            if user_act and ('[bye]' in user_act or '[thank]' in user_act):
                dial_gen['terminate_reason'] = 'user said thank or bye'
                return False
            # 不满足退出条件则继续循环
            return True

        while is_continue(dial_gen): # 需要判断一个会话是否结束，满足结束条件则需要退出循环
            if utterance_count & 1:
                '''
                system agent:
                input1: bs_prefix + context
                output1: belief states;
                input2: da_prefix + context + db_result
                output2: system action
                update user's goal state;
                input3: nlg_prefix + context + db_result
                output3: system response
                '''
                utterance_count += 1

                if user_utterance is None:
                    raise Exception('Should generate user utterance first!')

                # replace special tokens
                user_utterance_ids = self.encode_text(user_utterance, self.dialog_tokenizer, special_tokens_map=mttod_to_pptod)
                encoded_dialog_history = [self.encode_text(text, self.dialog_tokenizer, special_tokens_map=mttod_to_pptod) for text in dialog_history]
                context = self.flatten_dial_history(encoded_dialog_history, len(user_utterance_ids) + len(self.bs_prefix_id) + 1, self.dialog_tokenizer.model_max_length)
                input_ids = self.tensorize([self.bs_prefix_id + [self.sos_context_token_id] + context + user_utterance_ids + [self.eos_context_token_id]])
                input_ids = input_ids.to(device)

                with torch.no_grad():
                    model_output = self.dialog_model.generate(
                        input_ids=input_ids,
                        eos_token_id=self.dialog_tokenizer.eos_token_id,
                        max_length=100,
                    )
                belief_states_output = model_output.cpu().numpy().tolist()
                bspn_gen, _ = self.finalize_bspn(belief_states_output[0])

                bspn_gen = self.dialog_tokenizer.decode(bspn_gen, clean_up_tokenization_spaces=False)
                single_turn['belief_states'] = bspn_gen

                if turn_domain is None:
                    raise Exception('Domain is empty')

                db_token = self.bspn_to_db_pointer(bspn_gen, turn_domain)
                dbpn_gen = self.encode_text(db_token, self.dialog_tokenizer, bos_token='<sos_d>', eos_token='<eos_d>')
                single_turn['dbpn'] = self.dialog_tokenizer.decode(dbpn_gen)

                #action generation
                context_for_action = self.flatten_dial_history(encoded_dialog_history, len(user_utterance_ids) + len(self.da_prefix_id) + len(dbpn_gen) + 1, self.dialog_tokenizer.model_max_length)
                input_ids_da = self.tensorize([self.da_prefix_id + [self.sos_context_token_id] + context_for_action + user_utterance_ids + [self.eos_context_token_id] + dbpn_gen])
                input_ids_da = input_ids_da.to(device)

                with torch.no_grad():
                    model_output = self.dialog_model.generate(
                        input_ids=input_ids_da,
                        eos_token_id=self.dialog_tokenizer.eos_token_id,
                        max_length=100,
                    )

                aspn_outputs = model_output.cpu().numpy().tolist()
                aspn_gen, _ = self.finalize_aspn(aspn_outputs[0])
                aspn_gen = self.dialog_tokenizer.decode(aspn_gen, clean_up_tokenization_spaces=False).split()
                system_act_dict = convert_generate_action_span_to_dict(aspn_gen[1:-1])
                goal_state_dict = update_goal_states_during_gen(goal_state_dict, system_act_dict, 'sys')
                single_turn['sys_act'] = ' '.join(aspn_gen[1:-1])

                #response generation
                context_for_resp = self.flatten_dial_history(encoded_dialog_history, len(user_utterance_ids) + len(self.nlg_prefix_id) + len(dbpn_gen) + 1, self.dialog_tokenizer.model_max_length)
                input_ids_resp = self.tensorize([self.nlg_prefix_id + [self.sos_context_token_id] + context_for_resp + user_utterance_ids + [self.eos_context_token_id] + dbpn_gen])
                input_ids_resp = input_ids_resp.to(device)

                with torch.no_grad():
                    model_output = self.dialog_model.generate(
                        input_ids=input_ids_resp,
                        eos_token_id=self.dialog_tokenizer.eos_token_id,
                        max_length=200,
                    )
                
                resp_outputs = model_output.cpu().numpy().tolist()
                resp_gen, _ = self.finalize_resp(resp_outputs[0])
                resp_gen = self.dialog_tokenizer.decode(resp_gen, clean_up_tokenization_spaces=False).split()
                single_turn['sys'] = ' '.join(resp_gen[1:-1])
                log.append(single_turn.copy())
                single_turn = {}

                # update dialog history
                dialog_history.append(user_utterance)
                dialog_history.append(resp_gen)
                user_utterance = None
                turn_domain = None

            else:
                '''
                user agent:
                input: dialog history + goal state span;
                output: user action + user utterance;
                update user's goal state;
                '''
                utterance_count += 1

                goal_state_span = convert_goal_dict_to_span(goal_state_dict)
                goal_state_ids = self.encode_text(goal_state_span, self.simulator_tokenizer, bos_token=definitions.BOS_GOAL_TOEKN, eos_token=definitions.EOS_GOAL_TOKEN)
                encoded_dialog_history = [self.encode_text(text, self.simulator_tokenizer, special_tokens_map=pptod_to_mttod) for text in dialog_history]
                context = self.flatten_dial_history(encoded_dialog_history, len(goal_state_ids), self.simulator_tokenizer.model_max_length)
                input_ids = self.tensorize([context + goal_state_ids + [self.simulator_tokenizer.eos_token_id]])
                input_ids = input_ids.to(device)

                with torch.no_grad():
                    model_output = self.simulator_model.generate(
                    input_ids=input_ids,
                    eos_token_id=self.simulator_tokenizer.eos_token_id,
                    max_length=100,
                )

                user_utterance_output = model_output.cpu().numpy().tolist()
                user_act, user_utterance, _, _ = split_user_act_and_resp(self.simulator_tokenizer, user_utterance_output[0])
                user_act = self.simulator_tokenizer.decode(user_act, clean_up_tokenization_spaces=False).split(' ')
                user_utterance = self.simulator_tokenizer.decode(user_utterance, clean_up_tokenization_spaces=False).split(' ')

                if len(user_act[1:-1]) == 0 or user_act[1][1:-1] == 'general':
                    turn_domain = ['[general]']
                elif user_act[1][1:-1] not in definitions.ALL_DOMAINS:
                    # raise Exception('Invalid domain token')
                    turn_domain = ['[general]']
                else:
                    turn_domain = [user_act[1]]

                # only add user utterance to history
                single_turn['user'] = ' '.join(user_utterance[1:-1])
                single_turn['user_act'] = ' '.join(user_act[1:-1])

                # update goal state
                user_act_dict = convert_generate_action_span_to_dict(user_act[1:-1])
                goal_state_dict = update_goal_states_during_gen(goal_state_dict, user_act_dict, 'user')

        dial_gen['log'] = log
        dial_gen['final_goal_state'] = convert_goal_dict_to_span(goal_state_dict)
        return dial_gen

    def generate_single_dialog_ubar(self, user_goal):
        self.simulator_model.to(device)
        self.dialog_model.to(device)

        # clear fail info and invalid/prev_invalid field
        for domain in user_goal['goal']:
            if 'fail_info' in user_goal['goal'][domain]:
                del user_goal['goal'][domain]['fail_info']
            if 'fail_book' in user_goal['goal'][domain]:
                del user_goal['goal'][domain]['fail_book']

            if 'book' in user_goal['goal'][domain]:
                if 'invalid' in user_goal['goal'][domain]['book']:
                    del user_goal['goal'][domain]['book']['invalid']
                if 'pre_invalid' in user_goal['goal'][domain]['book']:
                    del user_goal['goal'][domain]['book']['pre_invalid']

        dial_gen = {user_goal['dialog_id']: {'goal': user_goal['goal']}}
        log = []
        dialog_history = []
        goal_state_dict = user_goal['goal']
        goal_state_span = convert_goal_dict_to_span(user_goal['goal'])
        user_utterance = None
        turn_domain = None
        system_act = None
        user_act = None
        utterance_count = 0
        single_turn = {}

        def is_continue(dial_gen):
            if 'sys' not in single_turn and 'user' in single_turn:
                # end up with system resp
                return True
            if len(goal_state_dict) == 0:
                # goal清空后终止
                dial_gen['terminate_reason'] = 'goal清空后终止'
                return False
            if len(log) >= self.cfg.max_turn_num:
                # 超过固定轮数终止
                dial_gen['terminate_reason'] = '超过{}轮终止'.format(self.cfg.max_turn_num)
                return False
            if system_act and ('[bye]' in system_act or '[thank]' in system_act):
                dial_gen['terminate_reason'] = 'system said thank or bye'
                return False
            if user_act and ('[bye]' in user_act or '[thank]' in user_act):
                dial_gen['terminate_reason'] = 'user said thank or bye'
                return False
            # 不满足退出条件则继续循环
            return True

        while is_continue(dial_gen): # 需要判断一个会话是否结束，满足结束条件则需要退出循环
            if utterance_count & 1:
                '''
                system agent:
                input1: context
                output1: belief states;
                input2: context + bs + db_result
                output2: system action
                update user's goal state;
                input3: context + bs + db_result + act
                output3: system response
                '''
                utterance_count += 1

                if user_utterance is None:
                    raise Exception('Should generate user utterance first!')

                # replace special tokens
                user_utterance_ids = self.encode_text(user_utterance, self.dialog_tokenizer, special_tokens_map=mttod_to_pptod)
                encoded_dialog_history = [self.encode_text(text, self.dialog_tokenizer, special_tokens_map=mttod_to_pptod) for text in dialog_history]
                context_for_bs = self.flatten_dial_history(encoded_dialog_history, len(user_utterance_ids) - 1 + 60, self.dialog_tokenizer.model_max_length)
                input_ids = self.tensorize([context_for_bs + user_utterance_ids])
                input_ids = input_ids.to(device)

                with torch.no_grad():
                    model_output = self.dialog_model.generate(
                        input_ids=input_ids,
                        pad_token_id=self.dialog_tokenizer.eos_token_id,
                        eos_token_id=self.dialog_tokenizer.encode(['<eos_b>'])[0],
                        max_length= input_ids.shape[1] + 60,
                        temperature=0.7,
                    )

                belief_states_output = model_output[:, input_ids.shape[1]:]
                belief_states_output = belief_states_output.cpu().numpy().tolist()
                bspn_gen, _ = self.finalize_bspn(belief_states_output[0])
                bspn_decoded = self.dialog_tokenizer.decode(bspn_gen, clean_up_tokenization_spaces=False)
                single_turn['belief_states'] = bspn_decoded

                if turn_domain is None:
                    raise Exception('Domain is empty')
                
                db_token = self.bspn_to_db_pointer(bspn_decoded, turn_domain)
                dbpn_gen = self.encode_text(db_token, self.dialog_tokenizer, bos_token='<sos_d>', eos_token='<eos_d>')
                dbpn_decoded = self.dialog_tokenizer.decode(dbpn_gen)
                single_turn['dbpn'] = dbpn_decoded

                #action generation
                context_for_da = self.flatten_dial_history(encoded_dialog_history, len(user_utterance_ids) + len(bspn_gen) + len(dbpn_gen) - 1 + 60, self.dialog_tokenizer.model_max_length)
                input_ids_da = self.tensorize([context_for_da + user_utterance_ids + bspn_gen + dbpn_gen])
                input_ids_da = input_ids_da.to(device)

                with torch.no_grad():
                    model_output = self.dialog_model.generate(
                        input_ids=input_ids_da,
                        pad_token_id=self.dialog_tokenizer.eos_token_id,
                        eos_token_id=self.dialog_tokenizer.encode(['<eos_a>'])[0],
                        max_length= input_ids_da.shape[1] + 60,
                        temperature=0.7,
                    )

                aspn_outputs = model_output[:, input_ids_da.shape[1]:]
                aspn_outputs = aspn_outputs.cpu().numpy().tolist()
                aspn_gen, _ = self.finalize_aspn(aspn_outputs[0])
                aspn_decoded = self.dialog_tokenizer.decode(aspn_gen, clean_up_tokenization_spaces=False).split()
                system_act_dict = convert_generate_action_span_to_dict(aspn_decoded[1:-1])
                goal_state_dict = update_goal_states_during_gen(goal_state_dict, system_act_dict, 'sys')
                single_turn['sys_act'] = ' '.join(aspn_decoded[1:-1])

                #response generation
                context_for_resp = self.flatten_dial_history(encoded_dialog_history, len(user_utterance_ids) + len(bspn_gen) + len(dbpn_gen) + len(aspn_gen) - 1 + 200, self.dialog_tokenizer.model_max_length)
                input_ids_resp = self.tensorize([context_for_resp + user_utterance_ids + bspn_gen + dbpn_gen + aspn_gen])
                input_ids_resp = input_ids_resp.to(device)

                with torch.no_grad():
                    model_output = self.dialog_model.generate(
                        input_ids=input_ids_resp,
                        pad_token_id=self.dialog_tokenizer.eos_token_id,
                        eos_token_id=self.dialog_tokenizer.encode(['<eos_r>'])[0],
                        max_length= input_ids_resp.shape[1] + 200,
                        temperature=0.7,
                    )

                resp_outputs = model_output[:, input_ids_resp.shape[1]:]
                resp_outputs = resp_outputs.cpu().numpy().tolist()
                resp_gen, _ = self.finalize_resp(resp_outputs[0])
                resp_decoded = self.dialog_tokenizer.decode(resp_gen, clean_up_tokenization_spaces=False).split()
                single_turn['sys'] = ' '.join(resp_decoded[1:-1])
                log.append(single_turn.copy())
                single_turn = {}

                prev_text = user_utterance + bspn_decoded.split() + dbpn_decoded.split() + aspn_decoded + resp_decoded
                dialog_history.append(prev_text)
                user_utterance = None
                turn_domain = None
            
            else:
                '''
                user agent:
                input: dialog history + goal state span;
                output: user action + user utterance;
                update user's goal state;
                '''
                utterance_count += 1

                goal_state_span = convert_goal_dict_to_span(goal_state_dict)
                goal_state_ids = self.encode_text(goal_state_span, self.simulator_tokenizer, bos_token=definitions.BOS_GOAL_TOEKN, eos_token=definitions.EOS_GOAL_TOKEN)
                encoded_dialog_history = [self.encode_text(text, self.simulator_tokenizer, special_tokens_map=pptod_to_mttod) for text in dialog_history]
                context = self.flatten_dial_history(encoded_dialog_history, len(goal_state_ids), self.simulator_tokenizer.model_max_length)
                input_ids = self.tensorize([context + goal_state_ids + [self.simulator_tokenizer.eos_token_id]])
                input_ids = input_ids.to(device)

                with torch.no_grad():
                    model_output = self.simulator_model.generate(
                    input_ids=input_ids,
                    eos_token_id=self.simulator_tokenizer.eos_token_id,
                    max_length=100,
                )

                user_utterance_output = model_output.cpu().numpy().tolist()
                user_act, user_utterance, _, _ = split_user_act_and_resp(self.simulator_tokenizer, user_utterance_output[0])
                user_act = self.simulator_tokenizer.decode(user_act, clean_up_tokenization_spaces=False).split(' ')
                user_utterance = self.simulator_tokenizer.decode(user_utterance, clean_up_tokenization_spaces=False).split(' ')

                if len(user_act[1:-1]) == 0 or user_act[1][1:-1] == 'general':
                    turn_domain = ['[general]']
                elif user_act[1][1:-1] not in definitions.ALL_DOMAINS:
                    # raise Exception('Invalid domain token')
                    turn_domain = ['[general]']
                else:
                    turn_domain = [user_act[1]]

                # only add user utterance to history
                single_turn['user'] = ' '.join(user_utterance[1:-1])
                single_turn['user_act'] = ' '.join(user_act[1:-1])

                # update goal state
                user_act_dict = convert_generate_action_span_to_dict(user_act[1:-1])
                goal_state_dict = update_goal_states_during_gen(goal_state_dict, user_act_dict, 'user')

        dial_gen['log'] = log
        dial_gen['final_goal_state'] = convert_goal_dict_to_span(goal_state_dict)
        return dial_gen

    def generate_single_dialog_galaxy(self, user_goal):
        pass
    
    def update_model(self, loss, agent):
        '''
        agent: sys or usr
        '''
        assert agent in ['sys', 'usr']
        loss.backward()
        if agent == 'sys':
            torch.nn.utils.clip_grad_norm_(self.dialog_model.parameters(), self.cfg.grad_clip)
            self.rl_sys_optimizer.step()
            self.rl_sys_optimizer.zero_grad()
        else:
            torch.nn.utils.clip_grad_norm_(self.simulator_model.parameters(), self.cfg.grad_clip)
            self.rl_usr_optimizer.step()
            self.rl_usr_optimizer.zero_grad()

    def get_rl_loss(self, gen_dial_batch, agent):
        '''
        agent: sys or usr
        '''
        assert agent in ['sys', 'usr']
        rl_loss = 0
        turn_num = 0
        
        for dial_id in gen_dial_batch:
            gen_dial = gen_dial_batch[dial_id]
            for turn in gen_dial:
                turn_rl_loss = 0
                if agent == 'sys':
                    prob = torch.cat((turn['bspn_prob'], turn['sys_act_resp_prob']))
                    assert prob.shape[0] == len(turn['sys_rewards'])
                    for i in range(len(prob)):
                        turn_rl_loss += -1 * torch.log(prob[i]) * turn['sys_rewards'][i]
                    if self.cfg.use_mean_rl_loss:
                        turn_rl_loss /= len(prob)
                elif agent == 'usr':
                    prob = turn['user_act_resp_prob']
                    assert prob.shape[0] == len(turn['usr_rewards'])
                    for i in range(len(prob)):
                        turn_rl_loss += -1 * torch.log(prob[i]) * turn['usr_rewards'][i]
                    if self.cfg.use_mean_rl_loss:
                        turn_rl_loss /= len(prob)
                rl_loss += turn_rl_loss
                turn_num += 1

        return rl_loss / turn_num
    
    def get_success_reward(self, gen_dial_batch, evaluator):
        '''
        assgining user rewards to turn['usr_rewards']
        assgining system rewards to turn['sys_rewards']
        '''
        batch_rewards = []
        for dial_id in gen_dial_batch:
            success, _ = evaluator.e2e_eval({dial_id: gen_dial_batch[dial_id]}, online_eval=True)
            if success > 0:
                reward = 1.0
            else:
                reward = 0.0
            
            # If a dialog is successful, we set the reward of each turn to 1
            self.all_rewards.append(reward)
            batch_rewards.append(reward)

            for turn in gen_dial_batch[dial_id]:
                usr_r, sys_r = reward, reward

                usr_rewards = []
                sys_rewards = []

                usr_len = len(turn['user_act_resp_prob'])
                sys_len = len(turn['bspn_prob']) + len(turn['sys_act_resp_prob'])

                for _ in range(usr_len):
                    usr_rewards.insert(0, usr_r)
                    usr_r = usr_r * self.cfg.discount_factor

                for _ in range(sys_len):
                    sys_rewards.insert(0, sys_r)
                    sys_r = sys_r * self.cfg.discount_factor

                turn['usr_rewards'] = usr_rewards
                turn['sys_rewards'] = sys_rewards

        return np.mean(batch_rewards)

    def get_ppl_reward(self, gen_dial_batch, evaluator, tokenizer):
        '''
        add user ppl rewards to turn['usr_rewards']
        add system ppl rewards to turn['sys_rewards']
        '''
        all_rewards = 0
        count_num = 0
        for dial_id in gen_dial_batch:
            for turn in gen_dial_batch[dial_id]:
                user_ids = torch.tensor([tokenizer.encode(turn['user']) + [tokenizer.eos_token_id]])
                resp_ids = torch.tensor([tokenizer.encode(turn['resp_gen']) + [tokenizer.eos_token_id]])
                user_ids = user_ids.to(device2)
                resp_ids = resp_ids.to(device2)

                with torch.no_grad():
                    user_outputs = evaluator(
                        input_ids=user_ids,
                        labels=user_ids,
                    )
                    resp_outputs = evaluator(
                        input_ids=resp_ids,
                        labels=resp_ids,
                    )

                user_loss = user_outputs.loss.cpu()
                resp_loss = resp_outputs.loss.cpu()

                user_ppl = torch.exp(user_loss)
                resp_ppl = torch.exp(resp_loss)
                if torch.isnan(user_ppl):
                    user_reward = 0
                else:
                    user_reward = 1 / (user_ppl ** 1)
                if torch.isnan(resp_ppl):
                    resp_reward = 0
                else:
                    resp_reward = 1 / (resp_ppl ** 1)

                all_rewards += user_reward + resp_reward
                count_num += 2

                usr_rewards = []
                sys_rewards = []

                usr_len = len(turn['user_act_resp_prob'])
                sys_len = len(turn['sys_act_resp_prob'])

                for _ in range(usr_len):
                    usr_rewards.insert(0, user_reward)
                    user_reward = user_reward * self.cfg.discount_factor
                for _ in range(sys_len):
                    sys_rewards.insert(0, resp_reward)
                    resp_reward = resp_reward * self.cfg.discount_factor

                for i in range(usr_len):
                    turn['usr_rewards'][i] += usr_rewards[i] * self.cfg.ppl_coef
                for i in range(sys_len):
                    turn['sys_rewards'][len(turn['bspn_prob']) + i] += sys_rewards[i] * self.cfg.ppl_coef
        
        return all_rewards / count_num

    def get_gpt_score_reward(self, gen_dial_batch, evaluator, tokenizer):
        '''
        add user gpt_score rewards to turn['usr_rewards']
        add system gpt_score rewards to turn['sys_rewards']
        '''
        all_rewards = 0
        count_num = 0
        for dial_id in gen_dial_batch:
            for turn in gen_dial_batch[dial_id]:
                user_ids = torch.tensor([tokenizer.encode(turn['user']) + [tokenizer.eos_token_id]])
                resp_ids = torch.tensor([tokenizer.encode(turn['resp_gen']) + [tokenizer.eos_token_id]])
                user_ids = user_ids.to(device2)
                resp_ids = resp_ids.to(device2)

                with torch.no_grad():
                    user_outputs = evaluator(
                        input_ids=user_ids,
                        labels=user_ids,
                    )
                    resp_outputs = evaluator(
                        input_ids=resp_ids,
                        labels=resp_ids,
                    )

                user_gpt_score = user_outputs.loss.cpu()
                resp_gpt_score = resp_outputs.loss.cpu()

                if torch.isnan(user_gpt_score):
                    user_reward = 0
                else:
                    user_reward = min(1, 1 / (user_gpt_score ** 1))
                if torch.isnan(resp_gpt_score):
                    resp_reward = 0
                else:
                    resp_reward = min(1, 1 / (resp_gpt_score ** 1))

                all_rewards += user_reward + resp_reward
                count_num += 2

                usr_rewards = []
                sys_rewards = []

                usr_len = len(turn['user_act_resp_prob'])
                sys_len = len(turn['sys_act_resp_prob'])

                for _ in range(usr_len):
                    usr_rewards.insert(0, user_reward)
                    user_reward = user_reward * self.cfg.discount_factor
                for _ in range(sys_len):
                    sys_rewards.insert(0, resp_reward)
                    resp_reward = resp_reward * self.cfg.discount_factor

                for i in range(usr_len):
                    turn['usr_rewards'][i] += usr_rewards[i] * self.cfg.gpt_score_coef
                for i in range(sys_len):
                    turn['sys_rewards'][len(turn['bspn_prob']) + i] += sys_rewards[i] * self.cfg.gpt_score_coef
        
        return all_rewards / count_num

    def get_nsp_score_reward(self, gen_dial_batch, evaluator, tokenizer):
        '''
        add user nsp rewards to turn['usr_rewards']
        add system nsp rewards to turn['sys_rewards']
        '''
        all_rewards = 0
        count_num = 0
        for dial_id in gen_dial_batch:
            single_dial = []
            for turn in gen_dial_batch[dial_id]:
                user_ids = tokenizer.encode(turn['user'])[1:-1]
                resp_ids = tokenizer.encode((turn['resp_gen']))[1:-1]
                single_dial.append(user_ids)
                single_dial.append(resp_ids)

            input_ids = []
            label_ids = []
            for i in range(1, len(single_dial)):
                input_ids.append(torch.tensor([tokenizer.cls_token_id] + single_dial[i - 1] + [tokenizer.sep_token_id] + single_dial[i]))
                label_ids.append(0)

            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
            label_ids = torch.tensor(label_ids)
            input_ids = input_ids.to(device2)
            label_ids = input_ids.to(device2)
            attention_mask = torch.where(input_ids == tokenizer.pad_token_id, 0, 1)

            with torch.no_grad():
                model_outputs = evaluator(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    label=label_ids,
                )
            softmax = nn.Softmax(dim=1)
            logits = model_outputs.logits
            logits = softmax(logits)
            nsp_score = logits[:, 0].sum()
            avg_nsp_score = nsp_score / input_ids.shape[0]

            for turn in gen_dial_batch[dial_id]:
                user_reward, resp_reward = avg_nsp_score, avg_nsp_score
                all_rewards += user_reward + resp_reward
                count_num +=  2
                usr_rewards = []
                sys_rewards = []

                usr_len = len(turn['user_act_resp_prob'])
                sys_len = len(turn['sys_act_resp_prob'])

                for _ in range(usr_len):
                    usr_rewards.insert(0, user_reward)
                    user_reward = user_reward * self.cfg.discount_factor
                for _ in range(sys_len):
                    sys_rewards.insert(0, resp_reward)
                    resp_reward = resp_reward * self.cfg.discount_factor

                for i in range(usr_len):
                    turn['usr_rewards'][i] += usr_rewards[i] * self.cfg.nsp_coef
                for i in range(sys_len):
                    turn['sys_rewards'][len(turn['bspn_prob']) + i] += sys_rewards[i] * self.cfg.nsp_coef

        return all_rewards / count_num
        

    def rl_validation(self, evaluator):
        dialogs_gen = []
        for goal in tqdm(self.goal_list['valid'][:500], desc='Validation'):
            dial_gen = interaction.generate_single_dialog(goal)
            dialogs_gen.append(dial_gen)
        dialogs_gen = convert_results_format(dialogs_gen)
        success, match = evaluator.e2e_eval(dialogs_gen, online_eval=True)
        return success, match


    def train_RL(self):
        self.all_rewards = [] # rewards container

        self.rl_sys_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.dialog_model.parameters()), lr=self.cfg.rl_lr)
        self.rl_usr_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.simulator_model.parameters()), lr=self.cfg.rl_lr)

        # create evaluator for get reward
        reader_cfg_path = os.path.join(self.cfg.model_dir, "run_config.json")
        reader_cfg = SimpleNamespace(**load_json(reader_cfg_path))
        reader = MultiWOZReader(reader_cfg, reader_cfg.version)
        evaluator = MultiWozEvaluator(reader, eval_data_type='train')
        evaluator_dev = MultiWozEvaluator(reader, eval_data_type='dev')

        # coherence reward
        if self.cfg.use_ppl_as_reward:
            ppl_model = GPT2LMHeadModel.from_pretrained(self.cfg.ppl_ckpt)
            ppl_model.to(device2)
            ppl_tokenizer = GPT2Tokenizer.from_pretrained(self.cfg.ppl_ckpt)
            logger.info('Load ppl model from {}'.format(self.cfg.ppl_ckpt))
            logger.info('Load ppl tokenizer from {}'.format(self.cfg.ppl_ckpt))
        
        if self.cfg.use_nsp_score_as_reward:
            nsp_score_model = BertForNextSentencePrediction.from_pretrained(self.cfg.nsp_ckpt)
            nsp_score_model.to(device2)
            nsp_score_tokenizer = BertTokenizer.from_pretrained(self.cfg.nsp_ckpt)
            logger.info('Load nsp model from {}'.format(self.cfg.nsp_ckpt))
            logger.info('Load nsp tokenizer from {}'.format(self.cfg.nsp_ckpt))

        if self.cfg.use_gpt_score_as_reward:
            gpt_score_model = GPT2LMHeadModel.from_pretrained(self.cfg.gpt_score_ckpt)
            gpt_score_model.to(device2)
            gpt_score_tokenizer = GPT2Tokenizer.from_pretrained(self.cfg.gpt_score_ckpt)
            logger.info('Load gpt score model from {}'.format(self.cfg.gpt_score_ckpt))
            logger.info('Load gpt score tokenizer from {}'.format(self.cfg.gpt_score_ckpt))

        best_success = 0
        best_success_epoch = 0
        random.shuffle(self.goal_list['valid'])

        for epoch in range(1, self.cfg.epochs + 1):
            self.cfg.rl_dial_one_epoch = min(len(self.goal_list['train']), self.cfg.rl_dial_one_epoch)
            n_batch = self.cfg.rl_dial_one_epoch // self.cfg.rl_batch_size
            # random.seed(epoch)
            random.shuffle(self.goal_list['train'])
            epoch_avg_rewards = 0
            epoch_avg_rl_loss = 0
            epoch_avg_ppl_rewards = 0
            epoch_avg_nsp_rewards = 0
            epoch_avg_gpt_score_rewards = 0

            # success, match = self.rl_validation(evaluator_dev)
            # logger.info('Before RL: Success rate: {}; Inform rate: {};'.format(success, match))

            for agent in ['usr', 'sys']:
                for i in tqdm(range(n_batch), desc='Reinforcement Learning ({})'.format(agent)):
                    
                    start_idx = i * self.cfg.rl_batch_size
                    end_idx = (i + 1) * self.cfg.rl_batch_size
                    dial_goals = self.goal_list['train'][start_idx:end_idx]

                    gen_dial_batch = []
                    for goal in dial_goals:
                        dial_gen = self.generate_single_dialog(goal, with_logprob=True, agent=agent)
                        gen_dial_batch.append(dial_gen)
                    gen_dial_batch = convert_results_format(gen_dial_batch)
                    avg_rewards = self.get_success_reward(gen_dial_batch, evaluator)
                    if self.cfg.use_ppl_as_reward:
                        ppl_reward = self.get_ppl_reward(gen_dial_batch, ppl_model, ppl_tokenizer)
                        epoch_avg_ppl_rewards += ppl_reward
                    if self.cfg.use_nsp_score_as_reward:
                        nsp_reward = self.get_nsp_score_reward(gen_dial_batch, nsp_score_model, nsp_score_tokenizer)
                        epoch_avg_nsp_rewards += nsp_reward
                    if self.cfg.use_gpt_score_as_reward:
                        gpt_score_reward = self.get_gpt_score_reward(gen_dial_batch, gpt_score_model, gpt_score_tokenizer)
                        epoch_avg_gpt_score_rewards += gpt_score_reward
                    rl_loss = self.get_rl_loss(gen_dial_batch, agent)
                    epoch_avg_rl_loss += rl_loss.item()
                    self.update_model(rl_loss, agent)

                    del rl_loss
                    del gen_dial_batch
                    torch.cuda.empty_cache()
                    epoch_avg_rewards += avg_rewards

            if self.cfg.use_nsp_score_as_reward and self.cfg.use_gpt_score_as_reward:
                logger.info('Epoch: {}; Avg success rewards: {}; Avg gpt score rewards: {}; Avg nsp rewards: {}; Avg RL Loss: {}'.format(epoch, epoch_avg_rewards / (2 * n_batch), epoch_avg_gpt_score_rewards / (2 * n_batch), epoch_avg_nsp_rewards / (2 * n_batch), epoch_avg_rl_loss / (2 * n_batch)))
            elif self.cfg.use_gpt_score_as_reward:
                logger.info('Epoch: {}; Avg success rewards: {}; Avg gpt score rewards: {}; Avg RL Loss: {}'.format(epoch, epoch_avg_rewards / (2 * n_batch), epoch_avg_gpt_score_rewards / (2 * n_batch), epoch_avg_rl_loss / (2 * n_batch)))
            elif self.cfg.use_nsp_score_as_reward:
                logger.info('Epoch: {}; Avg success rewards: {}; Avg nsp rewards: {}; Avg RL Loss: {}'.format(epoch, epoch_avg_rewards / (2 * n_batch), epoch_avg_nsp_rewards / (2 * n_batch), epoch_avg_rl_loss / (2 * n_batch)))
            else:
                logger.info('Epoch: {}; Avg rewards: {}; Avg RL Loss: {}'.format(epoch, epoch_avg_rewards / (2 * n_batch), epoch_avg_rl_loss / (2 * n_batch)))

            success, match = self.rl_validation(evaluator_dev)
            if success > best_success:
                best_success = success
                best_success_epoch = epoch
                simulator_dir = os.path.dirname(self.cfg.simulator_path)
                dialog_dir = os.path.dirname(self.cfg.dialog_sys_path)
                self.simulator_model.save_pretrained(os.path.join(simulator_dir, self.cfg.simulator_save_path + '_epoch_{}'.format(epoch)))
                self.simulator_tokenizer.save_pretrained(os.path.join(simulator_dir, self.cfg.simulator_save_path + '_epoch_{}'.format(epoch)))
                self.dialog_model.save_pretrained(os.path.join(dialog_dir, self.cfg.dialog_save_path + '_epoch_{}'.format(epoch)))
                self.dialog_tokenizer.save_pretrained(os.path.join(dialog_dir, self.cfg.dialog_save_path + '_epoch_{}'.format(epoch)))
            
            logger.info('Epoch: {}; Success rate: {}; Inform rate: {}; Best success: {}; Best epoch: {}'.format(epoch, success, match, best_success, best_success_epoch))

if __name__ == '__main__':
    '''
    沉默着，走了有，多遥远
    蓦然间，抬起头，才发现
    '''
    cfg = get_config()

    # different tod model
    if cfg.model_name == 'pptod':
        cfg.dialog_sys_path = './pptod_small_finetune/ckpt-epoch10'
    elif cfg.model_name == 'ubar':
        cfg.dialog_sys_path = './distilgpt2_finetune/ckpt-epoch35'

    interaction = InteractionEnvironment(cfg)
    dialogs_gen = []
    if cfg.do_rl_training:
        # random seeds
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

        logger.info("Set random seed to %d", cfg.seed)

        interaction.train_RL()
    else:
        count = 0
        for goal in tqdm(interaction.all_goals['test']):
            if cfg.model_name == 'mttod':
                dial_gen = interaction.generate_single_dialog(goal)
            elif cfg.model_name == 'pptod':
                dial_gen = interaction.generate_single_dialog_pptod(goal)
            elif cfg.model_name == 'ubar':
                dial_gen = interaction.generate_single_dialog_ubar(goal)
            elif cfg.model_name == 'galaxy':
                dial_gen = interaction.generate_single_dialog_galaxy(goal)
            dialogs_gen.append(dial_gen)
    
        save_json(dialogs_gen, cfg.generate_results_path)