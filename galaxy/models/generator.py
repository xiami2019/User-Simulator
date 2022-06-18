"""
Generator class.
"""

import math
import torch
import numpy as np

from galaxy.args import str2bool


def repeat(var, times):
    if isinstance(var, list):
        return [repeat(x, times) for x in var]
    elif isinstance(var, dict):
        return {k: repeat(v, times) for k, v in var.items()}
    elif isinstance(var, torch.Tensor):
        var = var.unsqueeze(1)
        expand_times = [1] * len(var.shape)
        expand_times[1] = times
        dtype = var.dtype
        var = var.float()
        var = var.repeat(*expand_times)
        shape = [var.shape[0] * var.shape[1]] + list(var.shape[2:])
        var = var.reshape(*shape)
        var = torch.tensor(var, dtype=dtype)
        return var
    else:
        return var


def gather(var, idx):
    if isinstance(var, list):
        return [gather(x, idx) for x in var]
    elif isinstance(var, dict):
        return {k: gather(v, idx) for k, v in var.items()}
    elif isinstance(var, torch.Tensor):
        out = var.index_select(dim=0, index=idx)
        return out
    else:
        return var


class Generator(object):
    """ Genrator class. """

    _registry = dict()

    @classmethod
    def register(cls, name):
        Generator._registry[name] = cls
        return

    @staticmethod
    def by_name(name):
        return Generator._registry[name]

    @staticmethod
    def create(hparams, *args, **kwargs):
        """ Create generator. """
        generator_cls = Generator.by_name(hparams.generator)
        return generator_cls(hparams, *args, **kwargs)

    @classmethod
    def add_cmdline_argument(cls, parser):
        group = parser.add_argument_group("Generator")
        group.add_argument("--generator", type=str, default="BeamSearch",
                           choices=["TopKSampling", "TopPSampling", "GreedySampling",
                                    "BeamSearch"])
        group.add_argument("--min_gen_len", type=int, default=1,
                           help="The minimum length of generated response.")
        group.add_argument("--max_gen_len", type=int, default=30,
                           help="The maximum length of generated response.")
        group.add_argument("--use_true_prev_bspn", type=str2bool, default=False, help="")
        group.add_argument("--use_true_prev_aspn", type=str2bool, default=False, help="")
        group.add_argument("--use_true_db_pointer", type=str2bool, default=False, help="")
        group.add_argument("--use_true_prev_resp", type=str2bool, default=False, help="")
        group.add_argument("--use_true_curr_bspn", type=str2bool, default=False, help="")
        group.add_argument("--use_true_curr_aspn", type=str2bool, default=False, help="")
        group.add_argument("--use_all_previous_context", type=str2bool, default=True, help="")
        group.add_argument("--use_true_bspn_for_ctr_eval", type=str2bool, default=False, help="")
        group.add_argument("--use_true_domain_for_ctr_eval", type=str2bool, default=False, help="")

        args, _ = parser.parse_known_args()
        generator_cls = cls.by_name(args.generator)
        generator_cls.add_cmdline_argument(group)
        return group

    def __init__(self, hparams, reader):
        self.vocab_size = reader.vocab_size
        self.bos_id = reader.bos_id
        self.eos_id = reader.eos_id
        self.unk_id = reader.unk_id
        self.pad_id = reader.pad_id
        self.min_gen_len = hparams.min_gen_len
        self.max_gen_len = hparams.max_gen_len
        self.use_gpu = hparams.use_gpu
        assert 1 <= self.min_gen_len <= self.max_gen_len
        return

    def __call__(self, step_fn, state):
        """
        Running generation.

        @param : step_fn : decoding one step
        @type : function

        @param : state : initial state
        @type : dict
        """
        raise NotImplementedError


class BeamSearch(Generator):
    """ BeamSearch generator. """

    @classmethod
    def add_cmdline_argument(cls, group):
        group.add_argument("--beam_size", type=int, default=5,
                           help="The beam size in beam search.")
        group.add_argument("--length_average", type=str2bool, default=False,
                           help="Whether to use length average.")
        group.add_argument("--length_penalty", type=float, default=-1.0,
                           help="The parameter(alpha) of length penalty.")
        group.add_argument("--ignore_unk", type=str2bool, default=True,
                           help="Whether to ignore unkown token in generation.")
        return group

    def __init__(self, hparams, reader):
        super().__init__(hparams, reader)
        self.beam_size = hparams.beam_size
        self.length_average = hparams.length_average
        self.length_penalty = hparams.length_penalty
        self.ignore_unk = hparams.ignore_unk
        return

    def __call__(self, step_fn, state, start_id=None, eos_id=None, max_gen_len=None, prev_input=None):
        """
        Running beam search.

        @param : step_fn : decoding one step
        @type : function

        @param : state : initial state
        @type : dict
        """
        if prev_input is not None:

            if isinstance(prev_input, list):
                length = max(list(map(lambda x: len(x), prev_input)))
                prev_input_numpy = np.full((len(prev_input), length), self.pad_id)
                for i, x in enumerate(prev_input):
                    prev_input_numpy[i, :len(x)] = x
                prev_input_tensor = torch.from_numpy(prev_input_numpy)
                if self.use_gpu:
                    prev_input_tensor = prev_input_tensor.cuda()

                for i in range(length):
                    state["pred_token"] = prev_input_tensor[:, i].unsqueeze(-1).unsqueeze(-1)
                    if i != 0:
                        state["pred_mask"] = torch.not_equal(state["pred_token"], self.pad_id).float()
                        state["pred_pos"] = state["pred_pos"] + state["pred_mask"].int()
                    _, state = step_fn(state)
            else:
                assert isinstance(prev_input, torch.Tensor)
                for i, input in enumerate(prev_input):
                    state["pred_token"] = input.expand(1, 1, 1)
                    if i != 0:
                        state["pred_mask"] = torch.not_equal(state["pred_token"], self.pad_id).float()
                        state["pred_pos"] = state["pred_pos"] + 1
                    _, state = step_fn(state)

        batch_size = state["batch_size"]
        beam_size = self.beam_size

        # shape: [batch_size, 1]
        pos_index = torch.arange(0, batch_size, 1, dtype=torch.int64) * beam_size
        pos_index = pos_index.unsqueeze(1)

        # shape: [batch_size, beam_size, 1]
        if start_id is None:
            start_id = self.bos_id
        if eos_id is None:
            eos_id = self.eos_id
        predictions = torch.ones([batch_size, beam_size, 1], dtype=torch.int64) * start_id

        if self.use_gpu:
            pos_index = pos_index.cuda()
            predictions = predictions.cuda()

        # initial input (start_id)
        state["pred_token"] = predictions[:, :1]
        if prev_input is not None:
            state["pred_mask"] = torch.not_equal(state["pred_token"], self.pad_id).float()
            state["pred_pos"] = state["pred_pos"] + 1

        # shape: [batch_size, vocab_size]
        scores, state = step_fn(state)

        unk_penalty = np.zeros(self.vocab_size, dtype="float32")
        unk_penalty[self.unk_id] = -1e10
        unk_penalty = torch.from_numpy(unk_penalty)

        eos_penalty = np.zeros(self.vocab_size, dtype="float32")
        eos_penalty[eos_id] = -1e10
        eos_penalty = torch.from_numpy(eos_penalty)

        scores_after_end = np.full(self.vocab_size, -1e10, dtype="float32")
        scores_after_end[self.pad_id] = 0  # 希望<eos>之后只生成<pad>，故使词表中log(p(<pad>))最高(0)
        scores_after_end = torch.from_numpy(scores_after_end)

        if self.use_gpu:
            unk_penalty = unk_penalty.cuda()
            eos_penalty = eos_penalty.cuda()
            scores_after_end = scores_after_end.cuda()

        if self.ignore_unk:
            scores = scores + unk_penalty
        scores = scores + eos_penalty

        # shape: [batch_size, beam_size]
        sequence_scores, preds = torch.topk(scores, self.beam_size)

        predictions = torch.cat([predictions, preds.unsqueeze(2)], dim=2)
        state = repeat(state, beam_size)

        parent_idx_list = []
        pred_list = []

        if max_gen_len is None:
            max_gen_len = self.max_gen_len
        for step in range(2, max_gen_len + 1):
            pre_ids = predictions[:, :, -1:]
            state["pred_token"] = pre_ids.reshape(batch_size * beam_size, 1, 1)
            state["pred_mask"] = torch.not_equal(state["pred_token"], self.pad_id).float()
            state["pred_pos"] = state["pred_pos"] + 1
            scores, state = step_fn(state)

            # Generate next
            # scores shape: [batch_size * beam_size, vocab_size]
            if self.ignore_unk:
                scores = scores + unk_penalty

            if step <= self.min_gen_len:
                scores = scores + eos_penalty

            # scores shape: [batch_size, beam_size, vocab_size]
            scores = scores.reshape(batch_size, beam_size, self.vocab_size)

            # previous token is [PAD] or [EOS]
            pre_eos_mask = (1 - torch.not_equal(pre_ids, eos_id).float()) + \
                           (1 - torch.not_equal(pre_ids, self.pad_id).float())

            scores = scores * (1 - pre_eos_mask) + \
                     pre_eos_mask.repeat(1, 1, self.vocab_size) * scores_after_end
            if self.length_average:
                scaled_value = pre_eos_mask + (1 - pre_eos_mask) * (1 - 1 / step)
                sequence_scores = sequence_scores.unsqueeze(2) * scaled_value
                scaled_value = pre_eos_mask + (1 - pre_eos_mask) * (1 / step)
                scores = scores * scaled_value
            elif self.length_penalty >= 0.0:
                scaled_value = pre_eos_mask + (1 - pre_eos_mask) * \
                    (math.pow((4 + step) / (5 + step), self.length_penalty))
                sequence_scores = scaled_value * sequence_scores
                scaled_value = pre_eos_mask + (1 - pre_eos_mask) * \
                    (math.pow(1 / (5 + step), self.length_penalty))
                scores = scores * scaled_value
            scores = scores + sequence_scores.unsqueeze(-1)
            scores = scores.reshape(batch_size, beam_size * self.vocab_size)

            topk_scores, topk_indices = torch.topk(scores, beam_size)
            # topk_indices: [batch_size, beam_size * self.vocab_size] (already reshaped)
            parent_idx = topk_indices.floor_divide(self.vocab_size)
            preds = topk_indices % self.vocab_size

            # Gather state / sequence_scores
            parent_idx = parent_idx + pos_index
            parent_idx = parent_idx.reshape(batch_size * beam_size)
            state = gather(state, parent_idx)
            sequence_scores = topk_scores

            predictions = predictions.reshape(batch_size * beam_size, step)
            predictions = gather(predictions, parent_idx)
            predictions = predictions.reshape(batch_size, beam_size, step)
            predictions = torch.cat([predictions, preds.unsqueeze(2)], dim=2)

        pre_ids = predictions[:, :, -1]
        pre_eos_mask = (1 - torch.not_equal(pre_ids, eos_id).float()) + \
                       (1 - torch.not_equal(pre_ids, self.pad_id).float())
        sequence_scores = sequence_scores * pre_eos_mask + (1 - pre_eos_mask) * (-1e10)

        indices = torch.argsort(sequence_scores, dim=1)
        indices = indices + pos_index
        indices = indices.reshape(-1)
        sequence_scores = sequence_scores.reshape(batch_size * beam_size)
        predictions = predictions.reshape(batch_size * beam_size, -1)
        sequence_scores = gather(sequence_scores, indices)
        predictions = gather(predictions, indices)
        sequence_scores = sequence_scores.reshape(batch_size, beam_size)
        predictions = predictions.reshape(batch_size, beam_size, -1)

        results = {
            "preds": predictions[:, -1],
            "scores": sequence_scores[:, -1]
        }
        return results


BeamSearch.register("BeamSearch")
