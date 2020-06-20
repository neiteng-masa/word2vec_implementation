
#ifndef PROGRESS_DISPLAY_HPP
#define PROGRESS_DISPLAY_HPP

#include <chrono>

class progress_display {
    public:
        progress_display(): progress_rate(0), whole_iter(0) {}
        void start() { 
            progress_rate = 0;
            whole_iter = 0;
            t_start = std::chrono::system_clock::now(); 
        }
        void report(double diff, int64_t iter) {
            progress_rate += diff;
            whole_iter += iter;

            auto t_now = std::chrono::system_clock::now();
            auto elap = std::chrono::duration_cast<std::chrono::seconds>(t_now - t_start).count() + 1e-2;
            printf("\r%d iter / sec | %.2f %% | %d sec          ", (int)(whole_iter / elap), progress_rate * 100, (int)((1 - progress_rate) / (progress_rate + 1e-8) * elap));
            fflush(stdout);
        }
        void finish() { printf("\n"); }
    private:
        std::chrono::system_clock::time_point t_start;
        double progress_rate;
        int64_t whole_iter;
};

#endif

