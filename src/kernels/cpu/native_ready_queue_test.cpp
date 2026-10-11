// Byte parity and sleep/wakeup stress for the opt-in NVFP4 dependency queue.
#include "strata/kernels/cpu/pool.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <thread>

namespace cpu = strata::kernels::cpu;
int main(int argc, char** argv) {
    const bool bench = argc == 2 && std::strcmp(argv[1], "--bench") == 0;
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt(cpu::kNvfp4Type, cpu::kNvfp4Type, cpu::H, cpu::FF, f, err)) return 2;
    f.fp32_activations = true;
    std::mt19937 rng(72419);
    std::vector<uint8_t> blob(f.bytes);
    for (size_t b = 0; b < f.tail_off; b += 36) {
        for (int j = 0; j < 4; ++j) blob[b + j] = 0x30 + rng() % 12;
        for (int j = 4; j < 36; ++j) blob[b + j] = (uint8_t) rng();
    }
    const float scales[4] = {0.0037f, 0.0021f, 0.0049f, 0};
    std::memcpy(blob.data() + f.tail_off, scales, sizeof scales);
    std::vector<float> x(cpu::MAXT * cpu::H);
    for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin((float) i * 0.013f);
    int checks = 0;
    for (int n : (bench ? std::vector<int>{1, 4, 8, 16} : std::vector<int>{1, 7, 97})) {
        std::vector<cpu::ExpertJobMulti> jobs(n);
        std::vector<float> out((size_t) n * cpu::MAXT * cpu::H, 0);
        for (int e = 0; e < n; ++e) {
            jobs[e].blob = blob.data();
            jobs[e].nt = bench ? 2 : 1 + e % cpu::MAXT;
            for (int t = 0; t < jobs[e].nt; ++t) {
                jobs[e].nact[t] = x.data() + t * cpu::H;
                jobs[e].out[t] = out.data() + ((size_t) e * cpu::MAXT + t) * cpu::H;
            }
        }
        cpu::ExpertPool baseline(bench ? 15 : 3, bench, true, cpu::PoolAffinity::All, 40);
        baseline.set_native_ready_queue(false);
        baseline.run_split_multi_native(f, jobs.data(), n);
        const auto ref = out;
        for (float v : ref) if (!std::isfinite(v)) return 1;
        if (bench) {
            std::vector<double> times[2];
            for (int round = 0; round < 12; ++round) {
                for (int j = 0; j < 2; ++j) {
                    const int arm = (round + j) % 2;
                    auto& pool = baseline;
                    pool.set_native_ready_queue(arm != 0);
                    const auto start = std::chrono::steady_clock::now();
                    for (int k = 0; k < 24; ++k) pool.run_split_multi_native(f, jobs.data(), n);
                    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / 24;
                    if (round >= 2) times[arm].push_back(ms);
                    if (std::memcmp(ref.data(), out.data(), out.size() * sizeof(float))) return 1;
                }
            }
            for (auto& arm : times) std::sort(arm.begin(), arm.end());
            std::printf("n=%d nt=2 original=%.4f ms ready=%.4f ms\n", n, times[0][5], times[1][5]);
            continue;
        }
        for (bool host : {false, true}) for (int workers : {1, 3}) for (int tasks : {1, 40, 128}) {
            cpu::ExpertPool pool(workers, false, host, cpu::PoolAffinity::All, tasks);
            pool.set_native_ready_queue(true);
            for (int repeat = 0; repeat < 3; ++repeat) {
                if (repeat == 1) std::this_thread::sleep_for(std::chrono::milliseconds(25));
                std::vector<cpu::ExpertPool::WarmBlob> warm;
                for (int e = 0; e < n; ++e) warm.push_back({jobs[e].blob, (size_t) f.bytes});
                warm.push_back({blob.data() + 1, 1});
                warm.push_back({blob.data() + 7, 65});
                pool.start_warm_ahead(warm.data(), (int) warm.size());
                if (repeat == 2) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(3));
                    // Replacing an outstanding batch retires it before publishing
                    // new metadata; run() also cancels without an explicit join.
                    pool.start_warm_ahead(warm.data(), (int) warm.size());
                }
                std::fill(out.begin(), out.end(), 0.f);
                for (const auto& job : jobs) for (int t = 0; t < job.nt; ++t)
                    std::fill(job.out[t], job.out[t] + cpu::H, std::numeric_limits<float>::quiet_NaN());
                pool.run_split_multi_native(f, jobs.data(), n);
                if (std::memcmp(ref.data(), out.data(), out.size() * sizeof(float))) {
                    std::fprintf(stderr, "mismatch n=%d host=%d workers=%d tasks=%d repeat=%d\n", n, host, workers, tasks, repeat);
                    return 1;
                }
                ++checks;
            }
        }
    }
    {
        cpu::ExpertPool pool(3, false);
        const cpu::ExpertPool::WarmBlob warm = {blob.data(), (size_t) f.bytes};
        pool.start_warm_ahead(&warm, 1);
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
        pool.finish_warm_ahead();
        if (!pool.warm_bytes()) return 3;
        pool.start_warm_ahead(&warm, 1);   // destruction cancels and joins
    }
    std::printf("native ready queue: %d byte comparisons passed\n", checks);
    return 0;
}
