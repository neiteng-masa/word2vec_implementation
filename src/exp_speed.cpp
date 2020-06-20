#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <math.h>

const int COUNT = 1e8;
int random_idx[COUNT];
double random_value[COUNT];
double exp_table[2000];

template<size_t TABLE_SIZE, int32_t EXP_MAX>
class sigmoid {
    private:
        double table[TABLE_SIZE];

    public:
        sigmoid() {
            for (int i = 0; i < TABLE_SIZE; i++) {
                double expx = exp((2.0 * i / TABLE_SIZE - 1) * EXP_MAX);
                table[i] = expx / (1 + expx);
            }
        }

        double operator() (double x) const {
            int idx = (int)((x / EXP_MAX + 1) / 2 * TABLE_SIZE);
            if (idx < 0) return 0;
            if (idx >= TABLE_SIZE) return 1;
            return table[idx];
        }
};

int main() {
    printf("%f\n", 1 / (1 + exp(100000)));
    for (int i = 0; i < 2000; i++) {
        double expx = exp((2.0 * i / 2000 - 1) * 10);
        exp_table[i] = expx / (1 + expx);
    }

    for (int k = 0; k < COUNT; k++) {
        random_value[k] = (rand() / ((double)RAND_MAX + 1) - 0.5) * 30;
    }

    sigmoid<2000, 10> f;
    auto st = std::chrono::system_clock::now();
    for (int k = 0; k < COUNT; k++) {
        double r = f(random_value[k]);
    }
    auto en = std::chrono::system_clock::now();
    auto elap = std::chrono::duration_cast<std::chrono::milliseconds>(en - st).count();
    printf("%d\n", (int)elap);

    st = std::chrono::system_clock::now();
    for (int k = 0; k < COUNT; k++) {
        int idx = (int)((random_value[k] / 10 + 1) / 2 * 2000);
        if (idx < 0 || idx >= 2000) continue;
        double r = exp_table[idx];
    }
    en = std::chrono::system_clock::now();
    elap = std::chrono::duration_cast<std::chrono::milliseconds>(en - st).count();
    printf("%d\n", (int)elap);

    st = std::chrono::system_clock::now();
    for (int k = 0; k < COUNT; k++) {
        double r = exp(random_value[k]);
    }
    en = std::chrono::system_clock::now();
    elap = std::chrono::duration_cast<std::chrono::milliseconds>(en - st).count();
    printf("%d\n", (int)elap);

    return 0;
}
