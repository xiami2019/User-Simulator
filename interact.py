# Using an user simulator interacting with a dialogue system to generate dialogue session
# input: user's goal, database, user simulator, dialogue system
# output: a session
import os
import math
import time
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from itertools import chain
from collections import OrderedDict
from types import SimpleNamespace
from transformers import T5ForConditionalGeneration, T5Tokenizer
from utils.utils import load_json, save_json, convert_goal_dict_to_span, convert_generate_action_span_to_dict, \
update_goal_states_during_gen, get_or_create_logger, split_user_act_and_resp
from utils import definitions
from external_knowledges import MultiWozDB
from evaluator import MultiWozEvaluator, convert_results_format
from reader import MultiWOZReader

logger = get_or_create_logger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_config():
    parser = argparse.ArgumentParser(description='RL config')
    parser.add_argument("-rl_dial_one_epoch", type=int, default=200)
    parser.add_argument("-rl_batch_size", type=int, default=1)
    parser.add_argument("-epochs", type=int, default=20)
    parser.add_argument("-simulator_path", type=str, default='./simulator_t5_small/simulator_rl_v4_epoch_3')
    parser.add_argument("-dialog_sys_path", type=str, default='./dialogue_t5_small/dialog_rl_v4_epoch_3')
    parser.add_argument("-data_dir", type=str, default='./data/MultiWOZ_2.0/')
    parser.add_argument("-model_dir", type=str, default="simulator_t5_small")
    parser.add_argument("-discount_factor", type=float, default=0.99)
    parser.add_argument('-rl_lr', type=float, default=0.0001, help='learning rate for reinforcement learning')
    parser.add_argument('-grad_clip', type=float, default=5)
    parser.add_argument("-seed", type=int, default=42)
    parser.add_argument('-do_rl_training', action="store_true")
    args = parser.parse_args()

    return args

class InteractionEnvironment(object):
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.simulator_model = self.load_model(self.cfg.simulator_path)
        self.dialog_model = self.load_model(self.cfg.dialog_sys_path)
        self.simulator_tokenizer = self.load_tokenizer(self.cfg.simulator_path)
        self.dialog_tokenizer = self.load_tokenizer(self.cfg.dialog_sys_path)
        self.data_dir = self.cfg.data_dir
        db_path = os.path.join(os.path.dirname(self.data_dir), 'db')
        logger.info("Load Database from {}".format(db_path))
        self.db = MultiWozDB(db_path)
        self.get_goal_list()

    @property
    def all_goals(self):
        return self.goal_list

    def load_model(self, model_path):
        logger.info("Load model from {}".format(model_path))
        if not os.path.exists(model_path):
            raise Exception('Model path is invalid!')
        return T5ForConditionalGeneration.from_pretrained(model_path)

    def load_tokenizer(self, tokenizer_path):
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

    def encode_text(self, text, tokenizer, bos_token=None, eos_token=None):
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
        eos_belief_token_id = self.dialog_tokenizer.convert_tokens_to_ids(definitions.EOS_BELIEF_TOKEN)
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

    def bspn_to_constraint_dict(self, bspn):
        bspn = bspn.split() if isinstance(bspn, str) else bspn

        constraint_dict = OrderedDict()
        domain, slot = None, None
        for token in bspn:
            if token == definitions.EOS_BELIEF_TOKEN:
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
            if len(log) >= 20:
                # 超过20轮终止
                dial_gen['terminate_reason'] = '超过20轮终止'
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
                
                # belief states generation
                torch.set_grad_enabled(if_sys_need_grad)
                model_output = dialog_generate(
                    self.dialog_model,
                    input_ids=input_ids,
                    eos_token_id=self.dialog_tokenizer.eos_token_id,
                    max_length=100,
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
                    max_length=200,
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
                    # assert len(system_act) == len(sys_act_prob)
                    # assert len(system_resp) == len(sys_resp_prob)
                    # single_turn['resp_prob'] = sys_resp_prob
                    # single_turn['sys_act_prob'] = sys_act_prob
                    single_turn['sys_act_resp_prob'] = resp_prob
                
                system_act = self.dialog_tokenizer.decode(system_act).split()
                system_resp = self.dialog_tokenizer.decode(system_resp).split()
                system_act_dict = convert_generate_action_span_to_dict(system_act[1:-1])
                goal_state_dict = update_goal_states_during_gen(goal_state_dict, system_act_dict, 'sys')
                
                single_turn['sys_act'] = ' '.join(system_act[1:-1])
                single_turn['sys'] = ' '.join(system_resp[1:-1])
                log.append(single_turn.copy())
                single_turn = {}

                # update dialog history
                dialog_history.append(user_utterance)
                dialog_history.append(system_resp)
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
                    max_length=200,
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
                    # assert len(user_act) == len(user_act_prob)
                    # assert len(user_utterance) == len(user_utterance_prob)
                    # single_turn['user_act_prob'] = user_act_prob
                    # single_turn['user_prob'] = user_utterance_prob
                    single_turn['user_act_resp_prob'] = user_utterance_prob

                user_act = self.simulator_tokenizer.decode(user_act).split(' ')
                user_utterance = self.simulator_tokenizer.decode(user_utterance).split(' ')

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
                    # prob = torch.cat((turn['bspn_prob'], turn['sys_act_prob'], turn['resp_prob']))
                    prob = torch.cat((turn['bspn_prob'], turn['sys_act_resp_prob']))
                    assert prob.shape[0] == len(turn['sys_rewards'])
                    for i in range(len(prob)):
                        turn_rl_loss += -1 * torch.log(prob[i]) * turn['sys_rewards'][i]
                elif agent == 'usr':
                    # prob = torch.cat((turn['user_prob'], turn['user_act_prob']))
                    prob = turn['user_act_resp_prob']
                    assert prob.shape[0] == len(turn['usr_rewards'])
                    for i in range(len(prob)):
                        turn_rl_loss += -1 * torch.log(prob[i]) * turn['usr_rewards'][i]
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

                # usr_len = len(turn['user_prob']) + len(turn['user_act_prob'])
                # sys_len = len(turn['bspn_prob']) + len(turn['sys_act_prob']) + len(turn['resp_prob'])

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
        best_success = 0
        best_success_epoch = 0
        random.shuffle(self.goal_list['valid'])

        for epoch in range(1, self.cfg.epochs + 1):
            self.cfg.rl_dial_one_epoch = min(len(self.goal_list['train']), self.cfg.rl_dial_one_epoch)
            n_batch = self.cfg.rl_dial_one_epoch // self.cfg.rl_batch_size
            random.seed(epoch)
            random.shuffle(self.goal_list['train'])
            epoch_avg_rewards = 0
            epoch_avg_rl_loss = 0

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
                    rl_loss = self.get_rl_loss(gen_dial_batch, agent)
                    epoch_avg_rl_loss += rl_loss.item()
                    self.update_model(rl_loss, agent)

                    # tqdm.write("update! rl_loss: {}".format(rl_loss.item()))
                    del rl_loss
                    del gen_dial_batch
                    torch.cuda.empty_cache()

                    epoch_avg_rewards += avg_rewards

            logger.info('Epoch: {}; Avg rewards: {}; Avg RL Loss: {}'.format(epoch, epoch_avg_rewards / (2 * n_batch), epoch_avg_rl_loss / (2 * n_batch)))

            success, match = self.rl_validation(evaluator_dev)
            if success > best_success:
                best_success = success
                best_success_epoch = epoch
                simulator_dir = os.path.dirname(self.cfg.simulator_path)
                dialog_dir = os.path.dirname(self.cfg.dialog_sys_path)
                self.simulator_model.save_pretrained(os.path.join(simulator_dir, 'simulator_rl_v4_epoch_{}'.format(epoch)))
                self.simulator_tokenizer.save_pretrained(os.path.join(simulator_dir, 'simulator_rl_v4_epoch_{}'.format(epoch)))
                self.dialog_model.save_pretrained(os.path.join(dialog_dir, 'dialog_rl_v4_epoch_{}'.format(epoch)))
                self.dialog_tokenizer.save_pretrained(os.path.join(dialog_dir, 'dialog_rl_v4_epoch_{}'.format(epoch)))
            
            logger.info('Epoch: {}; Success rate: {}; Inform rate: {}; Best success: {}; Best epoch: {}'.format(epoch, success, match, best_success, best_success_epoch))

if __name__ == '__main__':
    cfg = get_config()
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
            dial_gen = interaction.generate_single_dialog(goal)
            dialogs_gen.append(dial_gen)
    
        save_json(dialogs_gen, 'generate_example_test_t5_small_rl_v4.json')