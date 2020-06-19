#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <math.h>

const int MAX_N_READ = 1000;

std::vector<int> count;
std::vector<int32_t> whole_corpus;
int whole_corpus_len;
int vocab_num;
int epoch_num;
int n_dim;
double word_samp_thre;

double* v1, v2;

double uni_rand() {
    return rand() / ((double)RAND_MAX + 1);
}

int train() {
    printf("Training ...\n");
    int32_t* corpus = new int32_t[whole_corpus_len];
    for (int epoch = 0; epoch < epoch_num; epoch++) {
        printf("epoch %d\n", epoch);

        int corpus_len = 0;
        for (int t = 0; t < whole_corpus_len; t++) {
            int word = whole_corpus[t];
            double prob = 1 - (sqrt(count[word] / (word_samp_thre * whole_corpus_len)) + 1) * (word_samp_thre * whole_corpus_len) / count[word];
            if (uni_rand() < prob) continue;
            corpus[corpus_len] = word;
            corpus_len++;
        }
        printf("%d\n", corpus_len);
    }
    delete[] corpus;
}

int load_words(FILE *fp) {
    fseek(fp, 0, SEEK_SET);
    int32_t word_seq[MAX_N_READ];

    int k = 0;
    printf("Loading corpus ...\n");
    while (1) {
        if (feof(fp)) break;
        int n_read = fread(word_seq, sizeof(int32_t), MAX_N_READ, fp);
        if (n_read < MAX_N_READ && ferror(fp)) {
            printf("input error.");
            return -1;
        }

        for (int i = 0; i < n_read; i++) {
            int32_t word = word_seq[i];
            if (count.size() <= word) {
                int size = count.size();
                count.resize(word + 1);
                std::fill(std::end(count) - (count.size() - size), std::end(count), 0);
            }
            count[word]++;
            whole_corpus.push_back(word);
        }
    }
    whole_corpus_len = whole_corpus.size();
    vocab_num = count.size();

    return 0;
}

int main(int argc, char* argv[]) {
    char *in_path = argv[1];

    n_dim = 100;
    epoch_num = 5;
    word_samp_thre = 1e-3;

    FILE* fp = fopen(in_path, "rb");
    if (fp == NULL) exit(EXIT_FAILURE);
    load_words(fp);
    fclose(fp);

    train();

    return 0;
}
