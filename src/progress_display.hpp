
#ifndef PROGRESS_DISPLAY_HPP
#define PROGRESS_DISPLAY_HPP

#include <chrono>
#include <map>

class progress_display {
    public:
        progress_display():
            t_start(std::chrono::system_clock::now()),
            first_report(true)
        {
        }
        void start()
        { 
            iter.clear();
            max_iter.clear();
            t_start = std::chrono::system_clock::now(); 
            first_report = true;
        }
        void report(int64_t a_iter, int64_t a_max_iter, int thread_id)
        {
            std::lock_guard<std::mutex> lock(mtx);

            iter[thread_id] = a_iter;
            max_iter[thread_id] = a_max_iter;

            auto t_now = std::chrono::system_clock::now();
            auto elap = std::chrono::duration_cast<std::chrono::seconds>(t_now - t_start).count();

            int64_t iter_sum = 0;
            int64_t max_iter_sum = 0;
            for (const auto& p : iter) {
                auto id = p.first;
                iter_sum += iter[id];
                max_iter_sum += max_iter[id];
            }
            double prg = iter_sum / (max_iter_sum + 1e-8);

            if (first_report) {
                first_report = false;
                printf("\n");
            }
            const int BAR_LEN = 30;
            char bar[BAR_LEN + 10];
            bar[0] = bar[BAR_LEN + 1] = '|';
            bar[BAR_LEN + 2] = '\0';
            for (int i = 0; i < BAR_LEN; i++) {
                if (1. * i / BAR_LEN < prg) bar[i + 1] = '#';
                else bar[i + 1] = '-';
            }
            printf("\033[1A\033[2K%s %6.2f %% | %10d iter / s | %d s / %d s \n", bar, prg * 100, (int)(iter_sum / (elap + 1e-2)), std::max(0, (int)((1 - prg) / (prg + 1e-8) * elap)), (int)elap);
        }
    private:
        std::map<int, int64_t> iter;
        std::map<int, int64_t> max_iter;
        std::chrono::system_clock::time_point t_start;
        std::mutex mtx;
        bool first_report;
};

#endif

