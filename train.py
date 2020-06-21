import argparse
import numpy as np
import subprocess

def run():
    parser = argparse.ArgumentParser()

    parser.add_argument("corpus", help="Path to corpus data. ")
    parser.add_argument("output", help="Path to file to save the resulting word vectors. ")

    parser.add_argument("-d", "--dim", help="Word vector dimensionality. ", default = 300, type = int)
    parser.add_argument("-t", "--thread-num", help="Number of threads. ", default = 4, type = int)

    parser.add_argument("-e", "--epoch", help="Number of epoch. ", default = 5, type = int)
    parser.add_argument("-s", "--samp-disc", help="Frequency threshold for deciding whether to discard a word. ", default = 1e-5, type = float)
    parser.add_argument("-l", "--lr", help="Initial learning rate. ", default = 0.025, type = float)
    parser.add_argument("-w", "--window", help="Window size. ", default = 5, type = int)
    parser.add_argument("-n", "--neg", help="Number of negative sampling. ", default = 5, type = int)
    parser.add_argument("-m", "--min-count", help="This will discard words that appear less than [ min-count ] times. ", default = 20, type = int)

    args = parser.parse_args()

    print("Setting configuration file ... ")
    corpus = np.fromfile(args.corpus, "int32")
    corpus_len = corpus.shape[0] + 10
    vocab_num = np.max(corpus) + 10

    conf_str = \
        f"#ifndef CONF_HPP\n" + \
        f"#define CONF_HPP\n" + \
        f"\n" + \
        f"const int64_t MAX_CORPUS_LEN = {corpus_len};\n" + \
        f"const int64_t MAX_VOCAB_NUM = {vocab_num};\n" + \
        f"const int N_DIM = {args.dim};\n" + \
        f"const int MAX_N_THREAD = {args.thread_num + 2};\n" + \
        f"\n" + \
        f"#endif\n"

    with open("src/conf.hpp", "w") as f:
        f.write(conf_str)

    print("Building executable file for training ... ")
    cmd = "g++ src/train.cpp -o bin/train -mcmodel=large -std=c++17 -pthread -Wall -O2 -mtune=native -march=native"
    print(cmd)
    subprocess.run(cmd, shell = True, check = True)

    cmd = f"./bin/train {args.corpus} {args.output} -e {args.epoch} -s {args.samp_disc} -l {args.lr} -w {args.window} -n {args.neg} -m {args.min_count} -t {args.thread_num}"
    print(cmd)
    subprocess.run(cmd, shell = True, check = True)

if __name__ == "__main__":
    run()

