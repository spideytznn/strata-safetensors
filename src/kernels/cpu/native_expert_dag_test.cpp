// Scheduling parity and a repeatable CPU microbenchmark, without model downloads.
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace cpu = strata::kernels::cpu;
int main() {
    if (!cpu::cpu_avx512_ok()) { std::puts("native expert DAG: skipped (AVX-512 unavailable)"); return 77; }
    cpu::NativeFmt f;
    std::string error;
    if (!cpu::native_fmt(cpu::kNvfp4Type, cpu::kNvfp4Type, cpu::H, cpu::FF, f, error)) return 1;
    f.fp32_activations = true;
    std::vector<uint8_t> blobs(f.bytes * 13);
    std::mt19937 rng(371);
    for (auto& v : blobs) v = uint8_t(rng());
    for (int e = 0; e < 13; ++e) {
        auto* b = blobs.data() + e * f.bytes;
        for (size_t off = 0; off < f.tail_off; off += 36)
            for (int j = 0; j < 4; ++j) b[off + j] = uint8_t(24 + rng() % 32);
        const float scales[4] = {.031f, .023f, .017f, 0};
        std::memcpy(b + f.tail_off, scales, sizeof(scales));
    }
    std::vector<float> x(cpu::MAXT * cpu::H);
    for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin(float(i) * .017f);
    int checks = 0;
    for (int n : {1, 5, 13, 97}) {
        std::vector<cpu::ExpertJobMulti> jobs(n);
        std::vector<float> out(n * cpu::MAXT * cpu::H, 0);
        for (int e = 0; e < n; ++e) {
            jobs[e].blob = blobs.data() + (e % 13) * f.bytes;
            jobs[e].nt = e % 3 == 0 ? 1 : e % 3 == 1 ? 2 : cpu::MAXT;
            for (int t = 0; t < jobs[e].nt; ++t) {
                jobs[e].nact[t] = x.data() + t * cpu::H;
                jobs[e].out[t] = out.data() + (e * cpu::MAXT + t) * cpu::H;
            }
        }
        {
            cpu::ExpertPool baseline(3, false);
            baseline.set_native_dag(false);
            baseline.run_split_multi_native(f, jobs.data(), n);
        }
        const auto ref = out;
        for (float v : ref) if (!std::isfinite(v)) return 2;
        for (int workers : {1, 7}) for (bool host : {false, true}) for (int tasks : {1, 40, 128}) {
            cpu::ExpertPool pool(workers, false, host, cpu::PoolAffinity::All, tasks);
            pool.set_native_dag(true);
            for (int repeat = 0; repeat < 3; ++repeat) {
                std::fill(out.begin(), out.end(), 0);
                pool.run_split_multi_native(f, jobs.data(), n);
                if (std::memcmp(ref.data(), out.data(), out.size() * sizeof(float))) {
                    std::fprintf(stderr, "DAG mismatch n=%d workers=%d host=%d tasks=%d\n", n, workers, host, tasks);
                    return 3;
                }
                ++checks;
            }
        }
        if (n <= 13) {
            const auto affinity = cpu::pin_current_thread(cpu::planned_host_core());
            cpu::ExpertPool pool(15, true, true, cpu::PoolAffinity::All, 40);
            for (bool dag : {false, true}) {
                pool.set_native_dag(dag);
                for (int i = 0; i < 5; ++i) pool.run_split_multi_native(f, jobs.data(), n);
                const auto start = std::chrono::steady_clock::now();
                for (int i = 0; i < 40; ++i) pool.run_split_multi_native(f, jobs.data(), n);
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / 40;
                std::printf("experts=%d dag=%d ms=%.5f\n", n, dag, ms);
            }
            cpu::restore_thread_affinity(affinity);
        }
    }
    std::printf("native expert DAG: %d bitwise comparisons passed\n", checks);
}
