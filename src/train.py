import numpy as np
import argparse
import pickle as pkl
import os
from multiprocessing import Pool
import multiprocessing as multi
from tqdm import tqdm
import datetime
import time
from scipy import interpolate

class DiscreteDist:
    def __init__(self, p):
        size = p.shape[0] * 10
        cum = np.cumsum(p)
        cum *= size / cum[-1]
        cum = np.r_[0, cum] # p.shape[0] + 1
        n = np.arange(p.shape[0] + 1)
        f = interpolate.interp1d(cum, n)
        dist = f(np.arange(size)).astype("int64")
        dist[dist < 0] = 0
        dist[dist >= size] = size - 1
        self.dist = dist

    def gen(self, size):
        return self.dist[(np.random.rand(size) * self.dist.shape[0]).astype("int64")]

def train(whole_corpus, dim, epoch_num, window_size, word_samp_th, neg_samp_num, lr):
    print(datetime.datetime.now())
    print("The length of corpus: {}".format(whole_corpus.shape[0]))

    print("Counting number of occurrences of the words. ")
    vocab, count = np.unique(whole_corpus, return_counts = True)
    vocab_num = vocab.shape[0]
    print(datetime.datetime.now())
    print("The size of vocabulary: {}".format(vocab_num))

    print("Discarding too rare words.")
    count[count < min_count] = 0
    word_count_seq = count[whole_corpus]
    mask = word_count_seq > 0
    whole_corpus = whole_corpus[mask] # 低頻度語は除去
    word_count_seq = word_count_seq[mask]

    whole_corpus_len = whole_corpus.shape[0]

    starting_lr = lr

    print(datetime.datetime.now())
    print("Initializing. ")
    # ベクトル の初期化
    v1 = (np.random.rand(vocab_num, dim) - 0.5) / dim
    v2 = np.zeros((vocab_num, dim))

    # negative sampling 分布
    distorted_tf = (count ** 0.75)
    neg_dist = DiscreteDist(distorted_tf)

    trained_word_count = 0
    for epoch in range(epoch_num):
        print(datetime.datetime.now())
        print("Stating epoch {} / {}.".format(epoch + 1, epoch_num))
        print("Discarding high-frequency words.")
        prob_discard = 1 - (np.sqrt(word_count_seq / (word_samp_th * whole_corpus_len)) + 1) * (word_samp_th * whole_corpus_len) / word_count_seq
        corpus = whole_corpus[np.random.rand(whole_corpus_len) < prob_discard]
        corpus_len = corpus.shape[0]

        print(datetime.datetime.now())
        print("Training. ")
        for t in tqdm(range(corpus_len)):
            left = t - window_size
            right = t + window_size
            if left < 0 or right >= corpus_len - 1:
                continue
            word = corpus[t]
            context = np.r_[corpus[left : t], corpus[t + 1: right + 1]]
            neg_rv = neg_dist.gen(context.shape[0] * neg_samp_num)
            v1_w = v1[word] # dim
            v2_c = v2[context, :] # context_num, dim
            v2_neg = v2[neg_rv, :].reshape(-1, neg_samp_num, dim) # context_num, neg_samp_num, dim

            w_c = np.dot(v2_c, v1_w) # context_num
            w_neg = np.dot(v2_neg, v1_w) # context_num, neg_samp_num

            sigma = lambda x: 1 / (1 + np.exp(-x))
            # context_num, dim
            d_w = sigma(-w_c)[:, np.newaxis] * v2_c - np.sum(sigma(w_neg)[:, :, np.newaxis] * v2_neg, axis = 1)
            # context_num, dim
            d_c = sigma(-w_c)[:, np.newaxis] * v1_w[np.newaxis, :]
            # context_num, neg_samp_num, dim
            d_neg = -sigma(w_neg)[:, :, np.newaxis] * v1_w[np.newaxis, np.newaxis, :]
            v1[word] += lr * np.sum(d_w, axis = 0)
            v2[context, :] += lr * d_c
            v2[neg.reshape(-1), :] += lr * d_neg.reshape(-1, dim)

            trained_word_count += 1
            lr = max(starting_lr * (1 - trained_word_count / (epoch_num * whole_corpus_len)), starting_lr * 1e-3)

    v1[count == 0, : ] = 0
    return v1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--corpus", help="Path to corpus data. ")
    parser.add_argument("-o", "--output", help="Path to file to save the resulting word vectors. ")

    args = parser.parse_args()

    min_count = 5
    window_size = 5
    word_samp_th = 1e-4
    dim = 100
    neg_samp_num = 5
    lr = 0.025

    epoch_num = 5

    print("Loading corpus. ")
    corpus = np.fromfile(args.corpus, dtype="int32")

    corpus = corpus[:22435478] # 1 / 100

    vec = train(corpus, dim, epoch_num, window_size, word_samp_th, neg_samp_num, lr)
    print("Saving word vector. ")
    vec.astype("float64").tofile(args.output)


