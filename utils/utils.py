"""
   MTTOD: utils/io_utils.py

   implements simple I/O utilities for serialized objects and
   logger definitions.

   Copyright 2021 ETRI LIRS, Yohan Lee. All rights reserved.

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
import copy
import json
import pickle
import logging

from . import definitions


def save_json(obj, save_path, indent=4):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def load_json(load_path, lower=True):
    with open(load_path, "r", encoding="utf-8") as f:
        obj = f.read()

        if lower:
            obj = obj.lower()

        return json.loads(obj)


def save_pickle(obj, save_path):
    with open(save_path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(load_path):
    with open(load_path, "rb") as f:
        return pickle.load(f)


def save_text(obj, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        for o in obj:
            f.write(o + "\n")


def load_text(load_path, lower=True):
    with open(load_path, "r", encoding="utf-8") as f:
        text = f.read()
        if lower:
            text = text.lower()
        return text.splitlines()


def get_or_create_logger(logger_name=None, log_dir=None):
    logger = logging.getLogger(logger_name)

    # check whether handler exists
    if len(logger.handlers) > 0:
        return logger

    # set default logging level
    logger.setLevel(logging.DEBUG)

    # define formatters
    stream_formatter = logging.Formatter(
        fmt="%(asctime)s  [%(levelname)s] %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S")

    file_formatter = logging.Formatter(
        fmt="%(asctime)s  [%(levelname)s] %(module)s; %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S")

    # define and add handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, "log"))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger

def convert_generate_action_span_to_dict(act_span):
    '''
    span format: [domain] [intent] slot_name1, slot_name2, slot_name3
    '''
    act_dict = {}
    current_domain = None
    current_intent = None
    for single_token in act_span:
        if single_token.startswith('[') and single_token.endswith(']'):
            # judge if a domain or intent token or not
            temp_token = single_token[1:-1]
            if temp_token in definitions.ALL_DOMAINS or temp_token == 'general': # domain token
                current_domain = temp_token
                act_dict[current_domain] = {}
            else: # intent token
                current_intent = temp_token
                if current_domain:
                    act_dict[current_domain][current_intent] = set()
                else:
                    raise Exception("It's an intent without a domain token.")
        else: # slot name
            if current_domain and current_intent:
                act_dict[current_domain][current_intent].add(single_token)
            else:
                raise Exception("An domain token and an intent token are expected.")

    for domain in act_dict:
        for intent in act_dict[domain]:
            act_dict[domain][intent] = list(act_dict[domain][intent])

    return act_dict

def update_goal_states_during_gen(goal_states, actions, type):
    '''
    update goal states dict only using slot name
    '''
    map_act_tokens_to_inform_goal_tokens = {
        'depart': 'departure',
        'dest': 'destination',
        'price': 'pricerange',
    }
    map_act_tokens_to_request_goal_tokens = {
        'post': 'postcode',
        'addr': 'address',
        'ref': 'reference',
        'fee': 'price',
        'ticket': 'price',
    }

    new_goal_states = copy.deepcopy(goal_states)
    if type == 'user':
        # update inform slot in goal
        for domain in actions:
            if domain not in goal_states:
                continue
            if 'inform' in actions[domain]:
                for slot_name in actions[domain]['inform']:
                    if slot_name in map_act_tokens_to_inform_goal_tokens:
                        slot_name =  map_act_tokens_to_inform_goal_tokens[slot_name]

                    for intent in ['info', 'book']: # fail info have been cleaned when generation dialogues；
                        if intent not in goal_states[domain]:
                            continue

                        if len(goal_states[domain][intent]) == 0: # clear empty field
                            new_goal_states[domain].pop(intent)
                            if len(new_goal_states[domain]):
                                new_goal_states.pop(domain)
                            continue
                        
                        for goal_slot_name in goal_states[domain][intent]:
                            if goal_slot_name == slot_name:
                                new_goal_states[domain][intent].pop(goal_slot_name)
                                if len(new_goal_states[domain][intent]) == 0:
                                    new_goal_states[domain].pop(intent)
                                    if len(new_goal_states[domain]) == 0:
                                        new_goal_states.pop(domain)
    elif type == 'sys':
        for domain in actions:
            if domain not in goal_states:
                continue
            for intent in actions[domain]:
                if intent in ['inform', 'offerbooked']:
                    for slot_name in actions[domain][intent]:
                        inform_slot_name = map_act_tokens_to_inform_goal_tokens[slot_name] if slot_name in map_act_tokens_to_inform_goal_tokens else slot_name
                        request_slot_name = map_act_tokens_to_request_goal_tokens[slot_name] if slot_name in map_act_tokens_to_request_goal_tokens else slot_name

                        if request_slot_name == 'price' and domain in ['hotel', 'restaurant']:
                            request_slot_name = 'pricerange'

                        # update inform slot in goal
                        for goal_intent in ['info', 'book']:
                            if goal_intent not in goal_states[domain] or goal_intent not in new_goal_states[domain]:
                                continue

                            # clear empty intents
                            if len(goal_states[domain][goal_intent]) == 0:
                                new_goal_states[domain].pop(goal_intent)
                                if len(new_goal_states[domain]):
                                    new_goal_states.pop(domain)
                                continue

                            for goal_slot_name in goal_states[domain][goal_intent]:
                                if goal_slot_name == inform_slot_name:
                                    if goal_slot_name in new_goal_states[domain][goal_intent]:
                                        new_goal_states[domain][goal_intent].pop(goal_slot_name)
                            # 只有弹出goal_slot_name后才会考虑弹出goal_intent和domain；
                            if goal_intent in new_goal_states[domain] and len(new_goal_states[domain][goal_intent]) == 0:
                                new_goal_states[domain].pop(goal_intent)
                        
                        # update request slot in goal
                        if 'reqt' not in goal_states[domain] or domain not in new_goal_states or 'reqt' not in new_goal_states[domain]:
                            continue

                        for goal_slot_name in goal_states[domain]['reqt']:
                            if goal_slot_name == request_slot_name:
                                if goal_slot_name in new_goal_states[domain]['reqt']:
                                    new_goal_states[domain]['reqt'].remove(goal_slot_name)
                        if 'reqt' in new_goal_states[domain] and len(new_goal_states[domain]['reqt']) == 0:
                            new_goal_states[domain].pop('reqt')

            if len(new_goal_states[domain]) == 0:
                new_goal_states.pop(domain)
    else:
        raise Exception('Invalid action type!')
    
    return new_goal_states

def update_goal_states(goal_states, actions, type):
        map_act_tokens_to_inform_goal_tokens = {
            'depart': 'departure',
            'dest': 'destination',
            'price': 'pricerange',
        }
        map_act_tokens_to_request_goal_tokens = {
            'post': 'postcode',
            'addr': 'address',
            'ref': 'reference',
            'fee': 'price',
            'ticket': 'price',
        }

        map_sysact_value_to_goal_value = {
            'southern part of town': 'south',
        }

        new_goal_states = copy.deepcopy(goal_states)
        if type == 'user':
            # update inform slot in goal
            for domain in actions:
                if domain not in goal_states:
                    continue
                if 'inform' in actions[domain]:
                    for slot_pairs in actions[domain]['inform']:
                        slot_name, slot_value = slot_pairs
                        
                        if slot_name in map_act_tokens_to_inform_goal_tokens:
                            slot_name =  map_act_tokens_to_inform_goal_tokens[slot_name]

                        for intent in ['info', 'fail_info', 'book', 'fail_book']:
                            if intent not in goal_states[domain]:
                                continue

                            if len(goal_states[domain][intent]) == 0:
                                new_goal_states[domain].pop(intent)
                                if len(new_goal_states[domain]):
                                    new_goal_states.pop(domain)
                                continue
                            
                            for goal_slot_name, goal_slot_value in goal_states[domain][intent].items():
                                if goal_slot_name == slot_name and goal_slot_value == slot_value:
                                    new_goal_states[domain][intent].pop(goal_slot_name)
                                    if len(new_goal_states[domain][intent]) == 0:
                                        new_goal_states[domain].pop(intent)
                                        if len(new_goal_states[domain]) == 0:
                                            new_goal_states.pop(domain)
        elif type == 'sys':
            for domain in actions:
                if domain not in goal_states:
                    continue
                for intent in actions[domain]:
                    if intent in ['inform', 'offerbooked']:
                        for slot_pair in actions[domain][intent]:
                            slot_name, slot_value = slot_pair
                            
                            inform_slot_name = map_act_tokens_to_inform_goal_tokens[slot_name] if slot_name in map_act_tokens_to_inform_goal_tokens else slot_name
                            request_slot_name = map_act_tokens_to_request_goal_tokens[slot_name] if slot_name in map_act_tokens_to_request_goal_tokens else slot_name

                            if request_slot_name == 'price' and domain in ['hotel', 'restaurant']:
                                request_slot_name = 'pricerange'

                            if slot_value in map_sysact_value_to_goal_value:
                                slot_value = map_sysact_value_to_goal_value[slot_value]
                            # update inform slot in goal

                            for goal_intent in ['info', 'fail_info', 'book', 'fail_book']:
                                if goal_intent not in goal_states[domain] or goal_intent not in new_goal_states[domain]:
                                    continue

                                # clear empty intents
                                if len(goal_states[domain][goal_intent]) == 0:
                                    new_goal_states[domain].pop(goal_intent)
                                    if len(new_goal_states[domain]):
                                        new_goal_states.pop(domain)
                                    continue

                                for goal_slot_name, goal_slot_value in goal_states[domain][goal_intent].items():
                                    if goal_slot_name == inform_slot_name and goal_slot_value == slot_value:
                                        if goal_slot_name in new_goal_states[domain][goal_intent]:
                                            new_goal_states[domain][goal_intent].pop(goal_slot_name)
                                # 只有弹出goal_slot_name后才会考虑弹出goal_intent和domain；
                                if goal_intent in new_goal_states[domain] and len(new_goal_states[domain][goal_intent]) == 0:
                                    new_goal_states[domain].pop(goal_intent)
                            
                            # update request slot in goal
                            if 'reqt' not in goal_states[domain] or domain not in new_goal_states or 'reqt' not in new_goal_states[domain]:
                                continue

                            for goal_slot_name in goal_states[domain]['reqt']:
                                if goal_slot_name == request_slot_name:
                                    if goal_slot_name in new_goal_states[domain]['reqt']:
                                        new_goal_states[domain]['reqt'].remove(goal_slot_name)
                            if 'reqt' in new_goal_states[domain] and len(new_goal_states[domain]['reqt']) == 0:
                                new_goal_states[domain].pop('reqt')

                if len(new_goal_states[domain]) == 0:
                    new_goal_states.pop(domain)
        else:
            raise Exception('Invalid action type!')
        
        return new_goal_states

def convert_goal_dict_to_span(goal_dict):
    goal_span = ''
    for domain in goal_dict:
        seened_slot_pair = set()
        has_inform = False
        goal_span += '[' + domain + '] '
        # add inform slot first
        for intent in goal_dict[domain]:
            if intent in ['info', 'fail_info']:
                if not has_inform:
                    goal_span += '[inform] '
                    has_inform = True
                for slot_name, slot_value in goal_dict[domain][intent].items():
                    temp_span = slot_name + ' ' + slot_value
                    # 去重
                    if temp_span in seened_slot_pair:
                        continue
                    else:
                        seened_slot_pair.add(temp_span)
                    
                    if isinstance(slot_value, list):
                        raise Exception('Multiple values')
                    goal_span += '[value_' + slot_name + '] '
                    goal_span += slot_value + ' '

        # add book slot 
        has_book = False
        for intent in goal_dict[domain]:
            if intent in ['book', 'fail_book']:
                if not has_book:
                    goal_span += '[book] '
                    has_book = True
                for slot_name, slot_value in goal_dict[domain][intent].items():
                    if slot_name in ['pre_invalid', 'invalid']:
                        continue
                    # 去重
                    temp_span = slot_name + ' ' + slot_value
                    if temp_span in seened_slot_pair:
                        continue
                    else:
                        seened_slot_pair.add(temp_span)

                    if isinstance(slot_value, list):
                        raise Exception('Multiple values')
                    goal_span += '[value_' + slot_name + '] '
                    goal_span += slot_value + ' '

        # add request slot
        if 'reqt' in goal_dict[domain]:
            goal_span += '[request] '
            for slot_name in goal_dict[domain]['reqt']:
                    goal_span += slot_name + ' '                    

    goal_span = goal_span[:-1] # remove last space

    return goal_span