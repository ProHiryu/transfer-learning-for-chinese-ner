import codecs
import random
import math
from collections import Counter
import numpy as np
import re
import json
import os
import pickle


UNKNOWN_CHAR = '<UNK>'


def load_wordvec(wordvec_path, id_to_word, vec_dim, old_embeddings):
    """Load word vectors from pre-trained file.
    """
    embeddings = old_embeddings
    word_vectors = {}
    for i, line in enumerate(codecs.open(wordvec_path, 'r', 'utf-8')):
        line = line.rstrip().split()
        if len(line) == vec_dim + 1:
            word_vectors[line[0]] = np.array([float(x) for x in line[1:]]).astype(np.float32)
    for i in range(len(id_to_word)):
        if id_to_word[i] in word_vectors:
            embeddings[i] = word_vectors[id_to_word[i]]
    return embeddings


def load_sentence(path):
    sentences = []
    sentence = []
    for line in codecs.open(path, 'r', 'utf-8'):
        line = line.rstrip()
        if not line:
            if sentence:
                sentences.append(sentence)
                sentence = []
        else:
            word = line.split(' ')
            try:
                assert len(word) == 2
                sentence.append(word)
            except:
                continue
    if sentence:
        sentences.append(sentence)
    return sentences


def char_mapping(sentences):
    chars = [c[0] for s in sentences for c in s]
    chars_counter = Counter(chars)
    sorted_chars = sorted(chars_counter.items(), key=lambda x: (-x[1], x[0]))
    id2char, _ = list(zip(*sorted_chars))
    id2char = list(id2char)
    id2char.append(UNKNOWN_CHAR)
    vocab_len = len(id2char)
    char2id = dict(zip(id2char, range(vocab_len)))
    char2id[UNKNOWN_CHAR] = vocab_len - 1
    return id2char, char2id


def tag_mapping(tag2label):
    tag2id = tag2label
    id2tag = [0 for _ in range(len(tag2id))]
    for k in tag2id:
        id2tag[tag2id[k]] = k
    return id2tag, tag2id


def preprocess_data(sentences, char2id, tag2id):
    data = []
    for s in sentences:
        string = [w[0] for w in s]
        chars = [char2id[w if w in char2id else UNKNOWN_CHAR] for w in string]
        tags = [tag2id[w[1]] for w in s]
        data.append((string, chars, tags))
    return data


class BatchManager(object):
    def __init__(self, data, batch_size):
        assert batch_size > 0
        self.batch_data = self._sort_and_pad(data, batch_size)
        self.batch_count = len(self.batch_data)

    def _sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))
        sorted_data = sorted(data, key=lambda x: len(x[0]))
        batchs = []
        for i in range(num_batch):
            batchs.append(self._pad_data(sorted_data[i * batch_size: (i+1) * batch_size]))
        return batchs

    def _pad_data(self, data):
        strings, chars, tags, lengths = [], [], [], []
        max_len = max([len(s[0]) for s in data])
        for line in data:
            s, c, t = line
            padding = [0] * (max_len - len(s))
            strings.append(s)
            chars.append(c + padding)
            tags.append(t + padding)
            lengths.append(len(s))
        return [strings, chars, tags, lengths]

    def iter_batch(self, shuffle=True):
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(len(self.batch_data)):
            yield self.batch_data[idx]


def get_sentence(train_path, test_path):
    train_sentences = load_sentence(train_path)
    test_sentences = load_sentence(test_path)
    
    return train_sentences, test_sentences

def get_tag2label_json(json_path):
    with open(json_path, 'rb') as f:
        tag2label = json.load(f)
    f.close()
    
    return tag2label

def get_transform(train_sentences, map_file, tag2label_path, transfer_tag2label_path):
    if not os.path.isfile(map_file):
        tag2label = get_tag2label_json(tag2label_path)
        transfer_tag2label = get_tag2label_json(transfer_tag2label_path)
        id2char, char2id = char_mapping(train_sentences)
        id2tag, tag2id = tag_mapping(tag2label)
        transfer_id2tag, transfer_tag2id = tag_mapping(transfer_tag2label)
        with open(map_file, "wb") as f:
            pickle.dump([char2id, id2char, tag2id, id2tag, transfer_tag2id, transfer_id2tag], f)
    else:
        with open(map_file, "rb") as f:
            char2id, id2char, tag2id, id2tag, transfer_tag2id, transfer_id2tag = pickle.load(f)

    return char2id, id2char, tag2id, id2tag, transfer_tag2id, transfer_id2tag