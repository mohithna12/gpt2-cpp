#include "threadpool.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <string>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

namespace gpt2 {

namespace {
// How long an idle thread spins before blocking. Covers the gap between the jobs of one
// forward pass; between tokens or passes, threads go to sleep.
constexpr auto kSpin = std::chrono::microseconds(100);

inline void cpu_relax() {
#if defined(__aarch64__)
    asm volatile("yield");
#elif defined(__x86_64__)
    asm volatile("pause");
#endif
}

// Spins until pred() or kSpin elapses; returns pred().
template <typename Pred>
bool spin_until(Pred pred) {
    const auto deadline = std::chrono::steady_clock::now() + kSpin;
    for (int i = 0;; ++i) {
        if (pred()) return true;
        cpu_relax();
        if ((i & 63) == 63 && std::chrono::steady_clock::now() > deadline) return pred();
    }
}
}  // namespace

ThreadPool::ThreadPool(int n_threads) {
    for (int i = 1; i < std::max(1, n_threads); ++i) workers_.emplace_back([this] { worker_loop(); });
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mu_);
        stop_ = true;
    }
    start_cv_.notify_all();
    for (std::thread& t : workers_) t.join();
}

void ThreadPool::run_tasks() {
    for (int64_t i; (i = next_.fetch_add(1, std::memory_order_relaxed)) < n_;) (*fn_)(i);
}

void ThreadPool::worker_loop() {
    uint64_t seen = 0;
    while (true) {
        const auto new_job = [&] { return generation_.load(std::memory_order_acquire) != seen; };
        if (!spin_until(new_job)) {
            std::unique_lock<std::mutex> lock(mu_);
            start_cv_.wait(lock, [&] { return stop_ || new_job(); });
            if (stop_) return;
        }
        seen = generation_.load(std::memory_order_acquire);
        run_tasks();
        if (busy_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            std::lock_guard<std::mutex> lock(mu_);  // pairs with the caller's blocking wait
            done_cv_.notify_one();
        }
    }
}

void ThreadPool::parallel_for(int64_t n, const std::function<void(int64_t)>& fn) {
    if (n <= 0) return;
    if (workers_.empty() || n == 1) {
        for (int64_t i = 0; i < n; ++i) fn(i);
        return;
    }
    fn_ = &fn;
    n_ = n;
    next_.store(0, std::memory_order_relaxed);
    busy_.store(static_cast<int>(workers_.size()), std::memory_order_relaxed);
    {
        // Publishing under the mutex means a worker between its predicate check and its
        // wait can't miss the notify.
        std::lock_guard<std::mutex> lock(mu_);
        generation_.fetch_add(1, std::memory_order_release);
    }
    start_cv_.notify_all();
    run_tasks();
    const auto done = [&] { return busy_.load(std::memory_order_acquire) == 0; };
    if (!spin_until(done)) {
        std::unique_lock<std::mutex> lock(mu_);
        done_cv_.wait(lock, done);
    }
    fn_ = nullptr;
}

int default_num_threads() {
    if (const char* env = std::getenv("GPT2_THREADS")) {
        const int n = std::atoi(env);
        if (n > 0) return n;
    }
#if defined(__APPLE__)
    // Performance cores only: efficiency cores are much slower, and a pass is only as fast as
    // its slowest share of work.
    int perf = 0;
    size_t len = sizeof(perf);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &perf, &len, nullptr, 0) == 0 && perf > 0)
        return perf;
#endif
    return std::max(1u, std::thread::hardware_concurrency());
}

namespace {
std::unique_ptr<ThreadPool>& pool_slot() {
    static std::unique_ptr<ThreadPool> p;
    return p;
}
}  // namespace

ThreadPool& pool() {
    std::unique_ptr<ThreadPool>& p = pool_slot();
    if (!p) p = std::make_unique<ThreadPool>(default_num_threads());
    return *p;
}

void set_num_threads(int n) { pool_slot() = std::make_unique<ThreadPool>(std::max(1, n)); }

}  // namespace gpt2
