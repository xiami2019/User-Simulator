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

def compute_success_and_inform_rate(args, cfg, data):
    reader = MultiWOZReader(cfg, cfg.version)
    evaluator = MultiWozEvaluator(reader, args.data_type)

    if args.eval_type == 'offline':
        bleu, success, match = evaluator.e2e_eval(
            data, eval_dial_list=None, add_auxiliary_task=cfg.add_auxiliary_task)
        score = 0.5 * (success + match) + bleu
        logger.info('Offline Evaluation: match: %2.2f; success: %2.2f; bleu: %2.2f; score: %.2f',
                match, success, bleu, score)
    elif args.eval_type == 'online':
        success, match = evaluator.e2e_eval(
                data, eval_dial_list=None, add_auxiliary_task=cfg.add_auxiliary_task, online_eval=True)
        logger.info('Online Evaluation: match: %2.2f; success: %2.2f;', match, success)

def compute_gptscore(args, data):
    cfg = get_config_without_unknown()

    cfg.backbone = 'gpt2'
    cfg.ckpt = './bart_score_gpt_lm_model_lr_1e_4/ckpt-epoch6'
    cfg.version = args.version
    cfg.ppl_level = 'bart_score'
    cfg.compute_for_single = True
    cfg.task = 'ppl'
    cfg.device = args.device

    reader = Lm_Reader(cfg)
    runner = LMRunner(cfg, reader)

    gptscore = runner.evaluation_for_single(data)

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

    original_data = load_json(args.output_result_path)
    if args.eval_type == 'online':
        data = convert_results_format(original_data)
    else:
        data = original_data

    logger.info('Compute all metrics for {}'.format(args.output_result_path))

    if args.use_inform_success:
        compute_success_and_inform_rate(args, cfg, data)

    if args.use_gptscore:
        compute_gptscore(args, data)

    if args.use_nspscore:
        compute_nsp_score(args, data)

    save_results_with_metrics(args, data)
    
