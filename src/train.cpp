#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <math.h>
#include <sys/stat.h>
#include <chrono>
#include <thread>
#include <mutex>
#include "xor128.hpp"
#include "configure.hpp"
#include "progress_display.hpp"

int32_t whole_corpus[MAX_CORPUS_LEN];
int64_t whole_corpus_len;
int32_t corpus[MAX_CORPUS_LEN];

int32_t count[MAX_VOCAB_NUM];
int32_t vocab_num;

int32_t neg_dist_table[NEG_DIST_TABLE_SIZE];

double v1[MAX_VOCAB_NUM][N_DIM];
double v2[MAX_VOCAB_NUM][N_DIM];
std::mutex v1_mtx[MAX_VOCAB_NUM];
std::mutex v2_mtx[MAX_VOCAB_NUM];

int n_thread;
int epoch_num;
int min_count;
double word_samp_thre;
double learning_rate;
int window_size;
int neg_samp_num;

double sigmoid(double x) noexcept {
    const double EXP_MAX = 10;
    if (x > EXP_MAX) return 1;
    if (x < -EXP_MAX) return 0;
    double expx = exp(x);
    return expx / (1 + expx);
}

int train() {
    printf("Initializing model parameters ... \n");

    xor128 rand_gen;
    for (int w = 0; w < vocab_num; w++) {
        for (int i = 0; i < N_DIM; i++) {
            v1[w][i] = (rand_gen() - 0.5) / N_DIM;
            v2[w][i] = 0;
        }
    }

    printf("Creating %d threads ...\n", n_thread);

    std::thread* threads[MAX_N_THREAD];
    int corpus_len_per_thread = whole_corpus_len / n_thread;

    std::mutex progress_mtx;
    progress_display prg_disp;
    bool progress_first = true;

    for (int k = 0; k < n_thread; k++) {
        threads[k] = new std::thread(
        [ &progress_mtx, &prg_disp, &progress_first ] (int64_t loc_begin, int64_t loc_len) {
            double dw[N_DIM];

            double starting_lr = learning_rate;
            double lr = starting_lr;

            int64_t trained_word_count = 0;
            int32_t* loc_corpus = corpus + loc_begin;

            xor128 rand_gen;

            for (int epoch = 0; epoch < epoch_num; epoch++) {
                int64_t corpus_len = 0;
                int64_t corpus_len_wo_disc = 0;
                for (int64_t t = loc_begin; t < loc_begin + loc_len; t++) {
                    int32_t word = whole_corpus[t];
                    if (count[word] < min_count) continue;
                    corpus_len_wo_disc++;

                    double prob = 1 - (sqrt(count[word] / (word_samp_thre * whole_corpus_len)) + 1) * (word_samp_thre * whole_corpus_len) / count[word];
                    if (rand_gen() < prob) continue;

                    loc_corpus[corpus_len] = word;
                    corpus_len++;
                }
                for (int64_t t = 0; t < corpus_len; t++) {
                    const int progress_interval = 3e4;
                    if (t % progress_interval == 0) {
                        std::lock_guard<std::mutex> lock(progress_mtx);
                        if (progress_first) {
                            prg_disp.start();
                            progress_first = false;
                        }
                        prg_disp.report((double)progress_interval / corpus_len / epoch_num / n_thread, progress_interval);
                    }

                    int64_t left = t - window_size;
                    int64_t right = t + window_size;
                    if (left < 0 || right >= corpus_len - 1) continue;

                    int32_t word = loc_corpus[t];
                    for (int64_t tc = left; tc <= right; tc++) {
                        if (tc == t) continue;
                        int32_t context = loc_corpus[tc];

                        double w_c = 0; for (int i = 0; i < N_DIM; i++) w_c += v1[word][i] * v2[context][i];
                        double w_c_sig = sigmoid(-w_c);

                        for (int i = 0; i < N_DIM; i++) dw[i] = w_c_sig * v2[context][i];

                        for (int l = 0; l < neg_samp_num; l++) {
                            int32_t neg = word;
                            while (neg == word) neg = neg_dist_table[(int64_t)(rand_gen() * NEG_DIST_TABLE_SIZE)];
                            double w_neg = 0; for (int i = 0; i < N_DIM; i++) w_neg += v1[word][i] * v2[neg][i];
                            double w_neg_sig = sigmoid(w_neg);
                            for (int i = 0; i < N_DIM; i++) dw[i] -= w_neg_sig * v2[neg][i];

                            v2_mtx[neg].lock();
                            for (int i = 0; i < N_DIM; i++) v2[neg][i] -= lr * w_neg_sig * v1[word][i];
                            v2_mtx[neg].unlock();
                        }

                        v1_mtx[word].lock();
                        for (int i = 0; i < N_DIM; i++) v1[word][i] += lr * dw[i];
                        v1_mtx[word].unlock();
                        v2_mtx[context].lock();
                        for (int i = 0; i < N_DIM; i++) v2[context][i] += lr * w_c_sig * v1[word][i];
                        v2_mtx[context].unlock();
                    }

                    trained_word_count++;
                    lr = std::max(starting_lr * (1 - (double)trained_word_count / (epoch_num * corpus_len_wo_disc)), starting_lr * 1e-3);
                }
            }
            return 0;
        }
        , corpus_len_per_thread * k, corpus_len_per_thread);
    }

    printf("Threads started. \n");
    for (int k = 0; k < n_thread; k++) threads[k]->join();
    prg_disp.finish();
    printf("Training is finished. \n");
    for (int k = 0; k < n_thread; k++) delete threads[k];

    return 0;
}

int count_words() {
    std::fill(count, count + MAX_VOCAB_NUM, 0);
    vocab_num = 0;
    for (int64_t t = 0; t < whole_corpus_len; t++) {
        int32_t word = whole_corpus[t];
        if (word >= MAX_VOCAB_NUM) exit(EXIT_FAILURE);
        count[word]++;
        vocab_num = std::max(vocab_num, word + 1);
    }
    return 0;
}

int init_neg_dist_table() {
    double power = 0.75;
    double Z = 0;
    for (int64_t w = 0; w < vocab_num; w++) Z += pow(count[w], power);
    double cum = 0;
    int64_t w = -1;
    for (int64_t i = 0; i < NEG_DIST_TABLE_SIZE; i++) {
        double p = (double)i / NEG_DIST_TABLE_SIZE;
        while (1) {
            if (p < cum / Z) break;
            if (w >= vocab_num - 1) break;
            w++;
            cum += pow(count[w], power);
        }
        neg_dist_table[i] = w;
    }
    return 0;
}

int64_t get_file_size(const char* path) {
    struct stat st;
    if (stat(path, &st)) return -1;
    return st.st_size;
}

int main(int argc, char* argv[]) {
    char *in_path = argv[1];
    char *out_path = argv[2];

    min_count = 10;
    epoch_num = 5;
    word_samp_thre = 1e-3;
    learning_rate = 0.025;
    window_size = 5;
    neg_samp_num = 10;
    n_thread = 24;

    FILE* fp = fopen(in_path, "rb");
    if (fp == NULL) exit(EXIT_FAILURE);
    int64_t file_size = get_file_size(in_path);
    whole_corpus_len = file_size / sizeof(int32_t) / 5;
    printf("The length of corpus: %lld\n", (long long)whole_corpus_len);
    if (whole_corpus_len > MAX_CORPUS_LEN) exit(EXIT_FAILURE);
    printf("Loading corpus ...\n");
    fread(whole_corpus, sizeof(int32_t), whole_corpus_len, fp);
    fclose(fp);

    printf("Counting number of occurrences of the words ... \n");
    count_words();
    printf("The size of vocabulary: %d\n", vocab_num);
    printf("Making negative sampling distribution table ... \n");
    init_neg_dist_table();

    train();

    fp = fopen(out_path, "wb");
    if (fp == NULL) exit(EXIT_FAILURE);
    printf("Saving word vectors ...\n");
    fwrite(v1[0], sizeof(double), vocab_num * N_DIM, fp);
    fclose(fp);

    return 0;
}
