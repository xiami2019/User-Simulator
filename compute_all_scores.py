import os
import torch
import random
import argparse
import numpy as np

from reader import MultiWOZReader
from types import SimpleNamespace
from utils.utils import load_json, save_json, get_or_create_logger
from evaluator import convert_results_format, MultiWozEvaluator
from train_lm import get_config_without_unknown, LMRunner, BertRunner
from lm_dataset import Lm_Reader, Bert_Reader

logger = get_or_create_logger(__name__)

def compute_avg_len(data):
    total_usr_turn = 0
    total_sys_turn = 0
    total_usr_tokens = 0
    total_sys_tokens = 0
    max_len = 0
    for dial_id in data:
        for turn in data[dial_id]:
            total_sys_turn += 1
            total_usr_turn += 1
            total_usr_tokens += len(turn['user'].split())
            total_sys_tokens += len(turn['resp_gen'].split())
            max_len = max(max_len, len(turn['user'].split()))
            max_len = max(max_len, len(turn['resp_gen'].split()))
    

    logger.info('Max len: {}; Avg len: {}; Avg usr len: {}; Avg sys len: {};'.format(max_len, (total_usr_tokens + total_sys_tokens) / (total_sys_turn + total_usr_turn), total_usr_tokens / total_usr_turn, total_sys_tokens / total_sys_turn))

def compute_success_and_inform_rate(args, cfg, data):
    reader = MultiWOZReader(cfg, cfg.version)
    evaluator = MultiWozEvaluator(reader, args.data_type)

    if args.eval_type == 'offline':
        bleu, success, match = evaluator.e2e_eval(
            data, eval_dial_list=None, add_auxiliary_task=cfg.add_auxiliary_task, add_success_rate=True)
        score = 0.5 * (success + match) + bleu
        logger.info('Offline Evaluation: match: %2.2f; success: %2.2f; bleu: %2.2f; score: %.2f',
                match, success, bleu, score)
    elif args.eval_type == 'online':
        success, match = evaluator.e2e_eval(
                data, eval_dial_list=None, add_auxiliary_task=cfg.add_auxiliary_task, online_eval=True, add_success_rate=True)
        logger.info('Online Evaluation: match: %2.2f; success: %2.2f;', match, success)

def compute_gptscore(args, data, lm_ckpt, agent=None):
    cfg = get_config_without_unknown()

    cfg.backbone = 'gpt2'
    cfg.ckpt = lm_ckpt
    cfg.version = args.version
    cfg.ppl_level = 'bart_score'
    cfg.compute_for_single = True
    cfg.task = 'ppl'
    cfg.device = args.device

    setattr(cfg, 'gpt_score_singe_side', args.gpt_score_singe_side)
    setattr(cfg, 'agent', agent)

    reader = Lm_Reader(cfg)
    runner = LMRunner(cfg, reader)

    gptscore = runner.evaluation_for_single(data, cfg.gpt_score_normalize)

    if cfg.gpt_score_singe_side:
        logger.info('Online Evaluation: GPT Score for %s: %f;',cfg.agent, gptscore)
    else:
        logger.info('Online Evaluation: GPT Score: %f;', gptscore)

def compute_nsp_score(args, data):
    cfg = get_config_without_unknown()

    cfg.backbone = 'bert-base-uncased'
    cfg.ckpt = './bert_nsp_model_lr_1e_5_1/ckpt-epoch9'
    cfg.version = args.version
    cfg.compute_for_single = True
    cfg.task = 'nsp'
    cfg.device = args.device

    reader = Bert_Reader(cfg)
    runner = BertRunner(cfg, reader)

    nspscore = runner.evaluation_for_single(data)

    logger.info('Online Evaluation: NSP Score: %f;', nspscore)

def save_results_with_metrics(args, data):
    save_path = args.output_result_path[:-5] + '_with_metrics.json'
    save_json(data, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument for evaluation")

    parser.add_argument("-data_type", type=str, default="test", choices=["dev", "test"])
    parser.add_argument("-eval_type", type=str, default='offline', choices=['offline', 'online'])
    parser.add_argument("-output_result_path", type=str, required=True)
    parser.add_argument("-config_dir", type=str, required=True)
    parser.add_argument("-use_inform_success", type=bool, default=True)
    parser.add_argument("-use_gptscore", type=bool, default=True)
    parser.add_argument("-use_nspscore", type=bool, default=True)
    parser.add_argument("-seed", type=int, default=42)
    parser.add_argument("-gpt_score_normalize", action='store_true')
    parser.add_argument("-gpt_score_singe_side", action='store_true')
    parser.add_argument("-test_model_name", type=str, default=None)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setattr(args, 'device', device)

    cfg_path = os.path.join(args.config_dir, "run_config.json")
    cfg = SimpleNamespace(**load_json(cfg_path))

    setattr(args, 'version', cfg.version)

    if args.test_model_name != None:
        cfg.model_name = args.test_model_name

    original_data = load_json(args.output_result_path)
    if args.eval_type == 'online':
        data = convert_results_format(original_data)
    else:
        data = original_data

    logger.info('Compute all metrics for {}'.format(args.output_result_path))

    compute_avg_len(data)

    if args.use_inform_success:
        compute_success_and_inform_rate(args, cfg, data)

    # if args.use_gptscore:
    #     if args.gpt_score_singe_side:
    #         lm_ckpt_sys = './bart_score_gpt_lm_model_lr_1e_4_sys_side/ckpt-epoch6'
    #         lm_ckpt_usr = './bart_score_gpt_lm_model_lr_1e_4_usr_side/ckpt-epoch6'
    #         compute_gptscore(args, data, lm_ckpt_sys, agent='sys')
    #         compute_gptscore(args, data, lm_ckpt_usr, agent='usr')
    #     else:
    #         lm_ckpt = './bart_score_gpt_lm_model_lr_1e_4/ckpt-epoch6'
    #         compute_gptscore(args, data, lm_ckpt)

    lm_ckpt_sys = './bart_score_gpt_lm_model_lr_1e_4_sys_side/ckpt-epoch6'
    lm_ckpt_usr = './bart_score_gpt_lm_model_lr_1e_4_usr_side/ckpt-epoch6'
    compute_gptscore(args, data, lm_ckpt_sys, agent='sys')
    compute_gptscore(args, data, lm_ckpt_usr, agent='usr')
    lm_ckpt = './bart_score_gpt_lm_model_lr_1e_4/ckpt-epoch6'
    args.gpt_score_singe_side = False
    compute_gptscore(args, data, lm_ckpt)

    if args.use_nspscore:
        compute_nsp_score(args, data)

    save_results_with_metrics(args, data)
    
