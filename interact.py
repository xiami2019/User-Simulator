# Using an user simulator interacting with a dialogue system to generate dialogue session
# input: user's goal, database, user simulator, dialogue system
# output: a session
import os
import torch
import numpy
from itertools import chain
from collections import OrderedDict
from transformers import T5ForConditionalGeneration, T5Tokenizer
from utils.utils import load_json, save_json, convert_goal_dict_to_span, convert_generate_action_span_to_dict, update_goal_states_during_gen, get_or_create_logger
from utils import definitions
from external_knowledges import MultiWozDB

logger = get_or_create_logger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class InteractionEnvironment(object):
    def __init__(self, simulator_path, dialog_sys_path, data_dir) -> None:
        self.simulator_model = self.load_model(simulator_path)
        self.dialog_model = self.load_model(dialog_sys_path)
        self.simulator_tokenizer = self.load_tokenizer(simulator_path)
        self.dialog_tokenizer = self.load_tokenizer(dialog_sys_path)
        self.data_dir = data_dir
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

    def split_user_act_and_resp(self, model_output):
        if model_output[0] == self.simulator_tokenizer.pad_token:
            model_output = model_output[1:]
        if model_output[-1] == self.simulator_tokenizer.eos_token:
            model_output = model_output[:-1]

        # user aspn
        bos_user_act_token = definitions.BOS_USER_ACTION_TOKEN
        eos_user_act_token = definitions.EOS_USER_ACTION_TOKEN
        if bos_user_act_token in model_output and eos_user_act_token in model_output:
            bos_user_act_idx = model_output.index(bos_user_act_token)
            eos_user_act_idx = model_output.index(eos_user_act_token)
            user_aspn = model_output[bos_user_act_idx:eos_user_act_idx + 1]
        else:
            user_aspn = [bos_user_act_token, eos_user_act_token]

        # user utterance
        bos_user_resp_token = definitions.BOS_USER_TOKEN
        eos_user_resp_token = definitions.EOS_USER_TOKEN
        if bos_user_resp_token in model_output and eos_user_resp_token in model_output:
            bos_user_resp_idx = model_output.index(bos_user_resp_token)
            eos_user_resp_idx = model_output.index(eos_user_resp_token)
            user_utterance = model_output[bos_user_resp_idx:eos_user_resp_idx + 1]
        else:
            user_utterance = [bos_user_resp_token, eos_user_resp_token]

        return user_aspn, user_utterance

    def split_system_act_and_resp(self, model_output):
        bos_act_token_id = self.dialog_tokenizer.convert_tokens_to_ids(definitions.BOS_ACTION_TOKEN)
        eos_act_token_id = self.dialog_tokenizer.convert_tokens_to_ids(definitions.EOS_ACTION_TOKEN)
        
        bos_resp_token_id = self.dialog_tokenizer.convert_tokens_to_ids(definitions.BOS_RESP_TOKEN)
        eos_resp_token_id = self.dialog_tokenizer.convert_tokens_to_ids(definitions.EOS_RESP_TOKEN)

        eos_db_token_id = self.dialog_tokenizer.convert_tokens_to_ids(definitions.EOS_DB_TOKEN)
        eos_token_id = self.dialog_tokenizer.eos_token_id

        if eos_token_id in model_output:
            eos_idx = model_output.index(eos_token_id)
            model_output = model_output[:eos_idx]

        # aspn
        if bos_act_token_id in model_output and eos_act_token_id in model_output:
            bos_action_idx = model_output.index(bos_act_token_id)
            eos_action_idx = model_output.index(eos_act_token_id)
            aspn = model_output[bos_action_idx:eos_action_idx + 1]
        else:
            aspn = [bos_act_token_id, eos_act_token_id]
        
        if bos_resp_token_id in model_output and eos_resp_token_id in model_output:
            bos_resp_token_idx = model_output.index(bos_resp_token_id)
            eos_resp_token_idx = model_output.index(eos_resp_token_id)
            resp = model_output[bos_resp_token_idx:eos_resp_token_idx+1]
        elif eos_act_token_id in model_output:
            eos_action_idx = len(model_output) - model_output[::-1].index(eos_act_token_id) - 1
            resp = model_output[eos_action_idx + 1:]
            if resp[-1] != eos_resp_token_id:
                resp.append(eos_resp_token_id)
            if resp[0] != bos_resp_token_id:
                resp = [bos_resp_token_id] + resp
        else:
            resp = [bos_resp_token_id, eos_resp_token_id]


        return aspn, resp

    def finalize_bspn(self, belief_outputs):
        eos_belief_token_id = self.dialog_tokenizer.convert_tokens_to_ids(definitions.EOS_BELIEF_TOKEN)
        if belief_outputs[0] == self.dialog_tokenizer.pad_token_id:
            belief_outputs = belief_outputs[1:]
        if belief_outputs[-1] == self.dialog_tokenizer.eos_token_id:
            belief_outputs = belief_outputs[:-1]

        if eos_belief_token_id not in belief_outputs:
            eos_idx = len(belief_outputs) - 1
        else:
            eos_idx = belief_outputs.index(eos_belief_token_id)

        return belief_outputs[:eos_idx+1]

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

    def generate_single_dialog(self, user_goal):
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

        def is_continue():
            if 'sys' not in single_turn and 'user' in single_turn:
                # end up with system resp
                return True
            if len(goal_state_dict) == 0:
                # goal清空后终止
                print('goal清空后终止')
                return False
            if len(log) >= 20:
                # 超过20轮终止
                print('超过20轮终止')
                return False
            if system_act and ('[bye]' in system_act or '[thank]' in system_act):
                print('thank or bye')
                return False
            if user_act and ('[bye]' in user_act or '[thank]' in user_act):
                print('thank or bye')
                return False
            # 不满足退出条件则继续循环
            return True

        while is_continue(): # 需要判断一个会话是否结束，满足结束条件则需要退出循环
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
                context = self.flatten_dial_history(dialog_history, len(user_utterance_ids), self.dialog_tokenizer.model_max_length)
                context = self.encode_text(context, self.dialog_tokenizer)
                input_ids = self.tensorize([context + user_utterance_ids + [self.dialog_tokenizer.eos_token_id]])
                input_ids = input_ids.to(device)
                # belief states generation
                with torch.no_grad():
                    belief_states_output = self.dialog_model.generate(
                        input_ids=input_ids,
                        eos_token_id=self.dialog_tokenizer.eos_token_id,
                        max_length=100,
                    )
                belief_states_output = belief_states_output.cpu().numpy().tolist()
                bspn_gen = self.finalize_bspn(belief_states_output[0])
                bspn_gen = self.dialog_tokenizer.decode(bspn_gen, clean_up_tokenization_spaces=False)
                single_turn['belief_states'] = bspn_gen
                
                if turn_domain is None:
                    raise Exception('Domain is empty')
                db_token = self.bspn_to_db_pointer(bspn_gen, turn_domain)
                dbpn_gen = self.encode_text(db_token, self.dialog_tokenizer, bos_token=definitions.BOS_DB_TOKEN, eos_token=definitions.EOS_DB_TOKEN)

                resp_decoder_input_ids = self.tensorize([dbpn_gen])
                resp_decoder_input_ids = resp_decoder_input_ids.to(device)
                # response generation
                with torch.no_grad():
                    resp_outputs = self.dialog_model.generate(
                        input_ids=input_ids,
                        decoder_input_ids=resp_decoder_input_ids,
                        eos_token_id=self.dialog_tokenizer.eos_token_id,
                        max_length=100,
                    )
                resp_outputs = resp_outputs.cpu().numpy().tolist()
                system_act, system_resp = self.split_system_act_and_resp(resp_outputs[0])
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
                context = self.flatten_dial_history(dialog_history, len(goal_state_ids), self.simulator_tokenizer.model_max_length)
                context = self.encode_text(context, self.simulator_tokenizer)
                input_ids = self.tensorize([context + goal_state_ids + [self.simulator_tokenizer.eos_token_id]])
                input_ids = input_ids.to(device)
                with torch.no_grad():
                    user_utterance_output = self.simulator_model.generate(
                        input_ids=input_ids,
                        eos_token_id=self.simulator_tokenizer.eos_token_id,
                        max_length=100,
                    )
                # output_tokens = self.simulator_tokenizer.convert_ids_to_tokens(user_utterance_output[0])
                output_tokens = self.simulator_tokenizer.decode(user_utterance_output[0]).split(' ')
                user_act, user_utterance = self.split_user_act_and_resp(output_tokens)

                if len(user_act[1:-1]) == 0:
                    turn_domain = ['[general]']
                elif user_act[1][1:-1] not in definitions.ALL_DOMAINS:
                    raise Exception('Invalid domain token')
                else:
                    turn_domain = [user_act[1]]
                
                # only add user utterance to history
                single_turn['user'] = ' '.join(user_utterance[1:-1])
                single_turn['user_act'] = ' '.join(user_act[1:-1])

                # update goal state
                user_act_dict = convert_generate_action_span_to_dict(user_act[1:-1])
                goal_state_dict = update_goal_states_during_gen(goal_state_dict, user_act_dict, 'user')

        dial_gen['log'] = log
        print('生成完成')
        save_json(dial_gen, 'generate_example2.json')

if __name__ == '__main__':
    simulator_path = './simulator_t5_base/ckpt-epoch10'
    dialog_sys_path = './dialogue_t5_base/ckpt-epoch10'
    data_dir = './data/MultiWOZ_2.0/'
    interaction = InteractionEnvironment(simulator_path, dialog_sys_path, data_dir)
    test_goal = interaction.all_goals['valid'][0]
    interaction.generate_single_dialog(test_goal)