import torch
import torchtext
import argparse
import json
import os
import pickle
import re

import numpy as np
# noinspection PyUnresolvedReferences
import scispacy
import spacy
from gensim.models import KeyedVectors

from torchtext import data
from torchtext import datasets
import random
import numpy as np
import torch.nn as nn
from torchtext.vocab import Vocab
import spacy
#from captum.attr import LayerIntegratedGradients, TokenReferenceBase, visualization
############"config
import json


import datetime
import sys

def config(path: str):
    config = json.loads(open(path, 'r').read())
    return config
############################


def log(msg):
    now = datetime.datetime.now()
    print(str(now) + ' ' + msg)
    sys.stdout.flush()


if __name__ == '__main__':
    log('Preprocessing...')

    cfg = config('config.json')
    print(json.dumps(cfg, indent=2))

    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocess_wordvectors', type=bool, default=cfg['preprocessing']['word_vectors']['doit'])
    parser.add_argument('--vocab_size', type=int, default=cfg['preprocessing']['word_vectors']['vocab_size'])
    parser.add_argument('--file_word2vec', type=str, default=cfg['preprocessing']['word_vectors']['file_word2vec'])
    parser.add_argument('--dir_vocab', type=str, default=cfg['preprocessing']['word_vectors']['dir_vocab'])

    parser.add_argument('--preprocess_pubmed', type=bool, default=cfg['preprocessing']['pubmed']['doit'])
    parser.add_argument('--file_train_text', type=str, default=cfg['preprocessing']['pubmed']['file_train_text'])
    parser.add_argument('--file_dev_text', type=str, default=cfg['preprocessing']['pubmed']['file_dev_text'])
    parser.add_argument('--to_lower', type=bool, default=cfg['preprocessing']['pubmed']['to_lower'])
    parser.add_argument('--language_model', type=str, default=cfg['preprocessing']['pubmed']['language_model'])

    args = parser.parse_args()

    ####################
    ### WORD VECTORS ###
    ####################
    if args.preprocess_wordvectors:
        log('Preprocessing word vectors...')

        # preprocessing step 1: build vocabulary
        VOCAB_SIZE = args.vocab_size

        #fasttext = KeyedVectors.load_word2vec_format(args.file_word2vec, limit=VOCAB_SIZE)
        fasttext = KeyedVectors.load_word2vec_format(args.file_word2vec, limit=VOCAB_SIZE, binary = True)

        word2vec = {}

        # for LRP (first layer) we need a lower and an upper bound
        lower_bound = float('inf')
        upper_bound = float('-inf')

        # we sum and average all vectors here, for the unknown token
        sum_of_vectors = None
        #change here 23.06.21 from ".vocab" to "key_to_index"
        for word in fasttext.key_to_index:
            word2vec[word] = np.reshape(fasttext[word], (1, -1))

            # for lrp (first layer) determine lower and upper bounds
            min_coeff = np.min(word2vec[word])
            max_coeff = np.max(word2vec[word])
            lower_bound = min_coeff if min_coeff < lower_bound else lower_bound
            upper_bound = max_coeff if max_coeff > upper_bound else upper_bound

            # sum word vectors
            if sum_of_vectors is not None:
                sum_of_vectors = sum_of_vectors + word2vec[word]
            else:
                # if this is the first word, init the sum of vectors
                sum_of_vectors = word2vec[word]

        # handle the unknown token vector
        sum_of_vectors /= VOCAB_SIZE  # normalize
        unk = '<###-unk-###>'
        word2vec[unk] = sum_of_vectors
        # note: padding token not needed, padding is performed in the course of a dataset transformation

        max_coeff = np.max(word2vec[unk])
        upper_bound = max_coeff if max_coeff > upper_bound else upper_bound
        min_coeff = np.min(word2vec[unk])
        lower_bound = lower_bound if min_coeff < lower_bound else lower_bound

        lower_bound = str(round(lower_bound, ndigits=5))
        upper_bound = str(round(upper_bound, ndigits=5))

        cfg['preprocessing']['word_vectors']['lower_bound'] = float(lower_bound)
        cfg['preprocessing']['word_vectors']['upper_bound'] = float(upper_bound)

        file_name = f'vocab_size_{VOCAB_SIZE}_min_{lower_bound}_max_{upper_bound}.p'

        path = os.path.join(args.dir_vocab, file_name)
        cfg['preprocessing']['word_vectors']['vocab'] = path

        # serialize the vocabulary and document the lower and upper bound in the name of the file
        pickle.dump(word2vec, open(path, 'wb'))

        log('Updated config.json')
        with open('config.json', 'w') as fin:
            fin.write(json.dumps(cfg, indent=2))

        log(f'Saved vocabulary of size {VOCAB_SIZE} w/ lower bound {lower_bound} and upper bound {upper_bound} to {file_name}.')
        log('...done preprocessing word vectors.')

