#pragma once
// Minimal persistent thread pool for the forward pass. parallel_for hands out task indices
// through an atomic counter, so faster cores simply take more tasks. Results never depend on
// the number of threads: callers split work so that each task writes disjoint outputs.

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace gpt2 {

class ThreadPool {
public:
    // n_threads counts the calling thread, so ThreadPool(1) runs everything inline.
    explicit ThreadPool(int n_threads);
    ~ThreadPool();
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    int size() const { return static_cast<int>(workers_.size()) + 1; }

    // Calls fn(i) for every i in [0, n) and returns when all calls are done. The caller takes
    // part. fn must not throw. Not reentrant: don't call parallel_for from inside fn.
    void parallel_for(int64_t n, const std::function<void(int64_t)>& fn);

private:
    void worker_loop();
    void run_tasks();

    std::vector<std::thread> workers_;
    std::mutex mu_;
    std::condition_variable start_cv_, done_cv_;
    const std::function<void(int64_t)>* fn_ = nullptr;
    int64_t n_ = 0;
    std::atomic<int64_t> next_{0};
    // Workers spin briefly on generation_ before sleeping, and the caller spins on busy_, so
    // the ~75 back-to-back jobs of a decode step don't each pay a condition-variable wake-up.
    std::atomic<uint64_t> generation_{0};  // bumped per job so workers see each job once
    std::atomic<int> busy_{0};             // workers still on the current job
    bool stop_ = false;                    // guarded by mu_
};

// Process-wide pool used by the ops. Default size: GPT2_THREADS if set, else the number of
// performance cores (macOS) or std::thread::hardware_concurrency().
ThreadPool& pool();
// Replaces the process-wide pool. Not safe while a forward pass is running.
void set_num_threads(int n);
int default_num_threads();

}  // namespace gpt2
