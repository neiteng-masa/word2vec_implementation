import numpy as np
import argparse
from load_shaped_corpus import load_shaped_corpus
import pickle as pkl
import os
from multiprocessing import Pool
import multiprocessing as multi

def train(whole_corpus):
    def train_process(process_id):

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--corpus", help="Path to corpus data. ")

    args = parser.parse_args()

    min_count = 5

    corpus = np.fromfile(args.corpus, dtype="int32")
    word_id, count = np.unique(corpus, return_counts = True)

    word_count_seq = count[corpus]
    corpus = corpus[word_count_seq < min_count] # 低頻度語は除去

    train(corpus)

