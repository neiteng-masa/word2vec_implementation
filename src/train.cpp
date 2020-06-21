#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <math.h>
#include <sys/stat.h>
#include <chrono>
#include <thread>
#include <mutex>
#include <unistd.h>
#include <string.h>
#include "xor128.hpp"
#include "conf.hpp"
#include "progress_display.hpp"

const int64_t NEG_DIST_TABLE_SIZE = MAX_VOCAB_NUM * 100;

int32_t whole_corpus[MAX_CORPUS_LEN];
int64_t whole_corpus_len;
int32_t corpus[MAX_CORPUS_LEN];

int32_t count[MAX_VOCAB_NUM];
int32_t vocab_num;

int32_t neg_dist_table[NEG_DIST_TABLE_SIZE];

double v1[MAX_VOCAB_NUM][N_DIM];
double v2[MAX_VOCAB_NUM][N_DIM];

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
    int64_t corpus_len_per_thread = whole_corpus_len / n_thread;

    progress_display prg_disp;

    for (int k = 0; k < n_thread; k++) {
        threads[k] = new std::thread(
        [ &prg_disp ] (int64_t loc_begin, int64_t loc_len, int thread_id) {
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
                    if (count[word] == 0) continue;
                    corpus_len_wo_disc++;

                    double prob = 1 - (sqrt(count[word] / (word_samp_thre * whole_corpus_len)) + 1) * (word_samp_thre * whole_corpus_len) / count[word];
                    if (rand_gen() < prob) continue;

                    loc_corpus[corpus_len] = word;
                    corpus_len++;
                }
                for (int64_t t = 0; t < corpus_len; t++) {
                    const int progress_interval = 2e4;
                    if (t % progress_interval == 0) {
                        double prg = (epoch + 1. * t / corpus_len) / epoch_num;
                        prg_disp.report(trained_word_count, (int64_t)trained_word_count / (prg + 1e-8), thread_id);
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
                        for (int i = 0; i < N_DIM; i++) v2[context][i] += lr * w_c_sig * v1[word][i];

                        for (int l = 0; l < neg_samp_num; l++) {
                            int32_t neg = word;
                            while (neg == word) neg = neg_dist_table[(int64_t)(rand_gen() * NEG_DIST_TABLE_SIZE)];
                            double w_neg = 0; for (int i = 0; i < N_DIM; i++) w_neg += v1[word][i] * v2[neg][i];
                            double w_neg_sig = sigmoid(w_neg);
                            for (int i = 0; i < N_DIM; i++) dw[i] -= w_neg_sig * v2[neg][i];
                            for (int i = 0; i < N_DIM; i++) v2[neg][i] -= lr * w_neg_sig * v1[word][i];
                        }
                        for (int i = 0; i < N_DIM; i++) v1[word][i] += lr * dw[i];
                    }

                    trained_word_count++;
                    lr = std::max(starting_lr * (1 - (double)trained_word_count / (epoch_num * corpus_len_wo_disc)), 1e-4);
                }
            }
            return 0;
        }
        , corpus_len_per_thread * k, corpus_len_per_thread, k);
    }

    printf("Threads started. \n");
    for (int k = 0; k < n_thread; k++) threads[k]->join();
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

int discard_infrequent_words() {
    for (int64_t w = 0; w < vocab_num; w++)
        if (count[w] < min_count)
            count[w] = 0;
    return 0;
}

int nullify_infrequent_word_vectors() {
    for (int64_t w = 0; w < vocab_num; w++)
        if (count[w] == 0)
            for (int i = 0; i < N_DIM; i++)
                v1[w][i] = 0;
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
    if (argc < 3) {
        printf("%s [ corpus ] [ output ]\n", argv[0]);
        return 0;
    }

    char *in_path = argv[1];
    char *out_path = argv[2];

    min_count = 30;
    epoch_num = 5;
    word_samp_thre = 1e-5;
    learning_rate = 0.025;
    window_size = 5;
    neg_samp_num = 5;
    n_thread = 16;

    int opt;
    while ((opt = getopt(argc, argv, "e:s:l:w:n:t:m:")) != -1) {
        switch(opt) {
            case 'e': {
                char* end;
                epoch_num = strtol(optarg, &end, 10);
                if (strlen(end) > 0) exit(EXIT_FAILURE);
                break;
            }
            case 's': {
                char* end;
                word_samp_thre = strtod(optarg, &end);
                if (strlen(end) > 0) exit(EXIT_FAILURE);
                break;
            }
            case 'l': {
                char* end;
                learning_rate = strtod(optarg, &end);
                if (strlen(end) > 0) exit(EXIT_FAILURE);
                break;
            }
            case 'w': {
                char* end;
                window_size = strtol(optarg, &end, 10);
                if (strlen(end) > 0) exit(EXIT_FAILURE);
                break;
            }
            case 'n': {
                char* end;
                neg_samp_num = strtol(optarg, &end, 10);
                if (strlen(end) > 0) exit(EXIT_FAILURE);
                break;
            }
            case 't': {
                char* end;
                n_thread = strtol(optarg, &end, 10);
                if (strlen(end) > 0) exit(EXIT_FAILURE);
                break;
            }
            case 'm': {
                char* end;
                min_count = strtol(optarg, &end, 10);
                if (strlen(end) > 0) exit(EXIT_FAILURE);
                break;
            }
        }
    }

    if (n_thread > MAX_N_THREAD) exit(EXIT_FAILURE);

    FILE* fp = fopen(in_path, "rb");
    if (fp == NULL) exit(EXIT_FAILURE);

    int64_t file_size = get_file_size(in_path);
    whole_corpus_len = file_size / sizeof(int32_t);
    printf("The length of corpus: %lld\n", (long long)whole_corpus_len);
    if (whole_corpus_len > MAX_CORPUS_LEN) exit(EXIT_FAILURE);

    printf("Loading corpus ...\n");
    fread(whole_corpus, sizeof(int32_t), whole_corpus_len, fp);

    fclose(fp);

    printf("Counting number of occurrences of the words ... \n");
    count_words();
    printf("The size of vocabulary: %d\n", vocab_num);

    printf("Discarding infrequent words ... \n");
    discard_infrequent_words();

    printf("Making negative sampling distribution table ... \n");
    init_neg_dist_table();

    train();

    printf("Saving word vectors ...\n");

    nullify_infrequent_word_vectors();

    fp = fopen(out_path, "wb");
    if (fp == NULL) exit(EXIT_FAILURE);
    fwrite(v1[0], sizeof(double), vocab_num * N_DIM, fp);
    fclose(fp);

    return 0;
}
