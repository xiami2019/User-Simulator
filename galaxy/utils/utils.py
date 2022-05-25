import logging
import json
import numpy as np
from collections import OrderedDict
from galaxy.utils import ontology


def clean_replace(s, r, t, forward=True, backward=False):
    def clean_replace_single(s, r, t, forward, backward, sidx=0):
        # idx = s[sidx:].find(r)
        idx = s.find(r)
        if idx == -1:
            return s, -1
        idx_r = idx + len(r)
        if backward:
            while idx > 0 and s[idx - 1]:
                idx -= 1
        elif idx > 0 and s[idx - 1] != ' ':
            return s, -1

        if forward:
            while idx_r < len(s) and (s[idx_r].isalpha() or s[idx_r].isdigit()):
                idx_r += 1
        elif idx_r != len(s) and (s[idx_r].isalpha() or s[idx_r].isdigit()):
            return s, -1
        return s[:idx] + t + s[idx_r:], idx_r

    # source, replace, target = s, r, t
    # count = 0
    sidx = 0
    while sidx != -1:
        s, sidx = clean_replace_single(s, r, t, forward, backward, sidx)
        # count += 1
        # print(s, sidx)
        # if count == 20:
        #     print(source, '\n', replace, '\n', target)
        #     quit()
    return s


def py2np(list):
    return np.array(list)


def write_dict(fn, dic):
    with open(fn, 'w') as f:
        json.dump(dic, f, indent=2)


def f1_score(label_list, pred_list):
    tp = len([t for t in pred_list if t in label_list])
    fp = max(0, len(pred_list) - tp)
    fn = max(0, len(label_list) - tp)
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    return f1


class CamKVRVocab(object):
    def __init__(self, vocab_size, special_tokens=[]):
        """
        vocab for camrest and kvret dataset
        """
        self.vocab_size = vocab_size
        self._idx2word = {}
        self._word2idx = {}
        self._freq_dict = {}
        self.special_tokens = special_tokens
        for w in self.special_tokens:
            self._absolute_add_word(w)

    def __len__(self):
        return len(self._idx2word)

    def _absolute_add_word(self, w):
        idx = len(self)
        self._idx2word[idx] = w
        self._word2idx[w] = idx

    def add_word(self, word):
        if word not in self._freq_dict:
            self._freq_dict[word] = 0
        self._freq_dict[word] += 1

    def has_word(self, word):
        return self._freq_dict.get(word)

    def _add_to_vocab(self, word):
        if word not in self._word2idx:
            idx = len(self._idx2word)
            self._idx2word[idx] = word
            self._word2idx[word] = idx

    def construct(self):
        l = sorted(self._freq_dict.keys(), key=lambda x: -self._freq_dict[x])
        if len(l) + len(self._idx2word) < self.vocab_size:
            logging.warning('actual vocabulary set smaller than that configured: {}/{}'
                            .format(len(l) + len(self._idx2word), self.vocab_size))
        for word in l:
            self._add_to_vocab(word)
            if len(self._idx2word) >= self.vocab_size:
                break

    def load_vocab(self, vocab_path):
        self._freq_dict = json.loads(open(vocab_path+'.freq.json', 'r').read())
        self._word2idx = json.loads(open(vocab_path+'.word2idx.json', 'r').read())
        self._idx2word = {}
        for w, idx in self._word2idx.items():
            self._idx2word[idx] = w
        self.vocab_size_true = len(self._idx2word)
        print('vocab file loaded from "'+vocab_path+'"')
        print('Vocabulary size: %d' % (self.vocab_size_true))

    def save_vocab(self, vocab_path):
        _freq_dict = OrderedDict(sorted(self._freq_dict.items(), key=lambda kv:kv[1], reverse=True))
        write_dict(vocab_path+'.word2idx.json', self._word2idx)
        write_dict(vocab_path+'.freq.json', _freq_dict)

    def sentence_encode(self, word_list):
        return [self.encode(_) for _ in word_list]

    def sentence_decode(self, index_list, eos=None):
        l = [self.decode(_) for _ in index_list]
        if not eos or eos not in l:
            return ' '.join(l)
        else:
            idx = l.index(eos)
            return ' '.join(l[:idx])

    def nl_decode(self, l, eos=None):
        return [self.sentence_decode(_, eos) + '\n' for _ in l]

    def encode(self, word):
        word = '<unk>' if word not in self._word2idx else word
        return self._word2idx[word]

    def decode(self, idx):
        if type(idx) is not int:
            idx = int(idx.item())
        return self._idx2word.get(idx, '<unk>')


class MultiWOZVocab(object):
    def __init__(self, vocab_size=0):
        """
        vocab for multiwoz dataset
        """
        self.vocab_size = vocab_size
        self.vocab_size_oov = 0  # get after construction
        self._idx2word = {}  # word + oov
        self._word2idx = {}  # word
        self._freq_dict = {}  # word + oov
        for w in ['[PAD]', '<go_r>', '[UNK]', '<go_b>', '<go_a>', '<eos_u>', '<eos_r>',
                  '<eos_b>', '<eos_a>', '<go_d>', '<eos_d>']:
            self._absolute_add_word(w)

    def _absolute_add_word(self, w):
        idx = len(self._idx2word)
        self._idx2word[idx] = w
        self._word2idx[w] = idx

    def add_word(self, word):
        if word not in self._freq_dict:
            self._freq_dict[word] = 0
        self._freq_dict[word] += 1

    def has_word(self, word):
        return self._freq_dict.get(word)

    def _add_to_vocab(self, word):
        if word not in self._word2idx:
            idx = len(self._idx2word)
            self._idx2word[idx] = word
            self._word2idx[word] = idx

    def construct(self):
        l = sorted(self._freq_dict.keys(), key=lambda x: -self._freq_dict[x])
        print('Vocabulary size including oov: %d' % (len(l) + len(self._idx2word)))
        if len(l) + len(self._idx2word) < self.vocab_size:
            logging.warning('actual label set smaller than that configured: {}/{}'
                            .format(len(l) + len(self._idx2word), self.vocab_size))
        for word in ontology.all_domains + ['general']:
            word = '[' + word + ']'
            self._add_to_vocab(word)
        for word in ontology.all_acts:
            word = '[' + word + ']'
            self._add_to_vocab(word)
        for word in ontology.all_slots:
            self._add_to_vocab(word)
        for word in l:
            if word.startswith('[value_') and word.endswith(']'):
                self._add_to_vocab(word)
        for word in l:
            self._add_to_vocab(word)
        self.vocab_size_oov = len(self._idx2word)

    def load_vocab(self, vocab_path):
        self._freq_dict = json.loads(open(vocab_path + '.freq.json', 'r').read())
        self._word2idx = json.loads(open(vocab_path + '.word2idx.json', 'r').read())
        self._idx2word = {}
        for w, idx in self._word2idx.items():
            self._idx2word[idx] = w
        self.vocab_size_oov = len(self._idx2word)
        print('vocab file loaded from "' + vocab_path + '"')
        print('Vocabulary size including oov: %d' % (self.vocab_size_oov))

    def save_vocab(self, vocab_path):
        _freq_dict = OrderedDict(sorted(self._freq_dict.items(), key=lambda kv: kv[1], reverse=True))
        write_dict(vocab_path + '.word2idx.json', self._word2idx)
        write_dict(vocab_path + '.freq.json', _freq_dict)

    def encode(self, word, include_oov=True):
        if include_oov:
            if self._word2idx.get(word, None) is None:
                raise ValueError('Unknown word: %s. Vocabulary should include oovs here.' % word)
            return self._word2idx[word]
        else:
            word = '<unk>' if word not in self._word2idx else word
            return self._word2idx[word]

    def sentence_encode(self, word_list):
        return [self.encode(_) for _ in word_list]

    def oov_idx_map(self, idx):
        return 2 if idx > self.vocab_size else idx

    def sentence_oov_map(self, index_list):
        return [self.oov_idx_map(_) for _ in index_list]

    def decode(self, idx, indicate_oov=False):
        if not self._idx2word.get(idx):
            raise ValueError('Error idx: %d. Vocabulary should include oovs here.' % idx)
        if not indicate_oov or idx < self.vocab_size:
            return self._idx2word[idx]
        else:
            return self._idx2word[idx] + '(o)'

    def sentence_decode(self, index_list, eos=None, indicate_oov=False):
        l = [self.decode(_, indicate_oov) for _ in index_list]
        if not eos or eos not in l:
            return ' '.join(l)
        else:
            idx = l.index(eos)
            return ' '.join(l[:idx])

    def nl_decode(self, l, eos=None):
        return [self.sentence_decode(_, eos) + '\n' for _ in l]
