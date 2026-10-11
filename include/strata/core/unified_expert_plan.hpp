#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace strata::core {

struct BatchExpertWork {
    int cpu_groups = 0, cpu_entries = 0;
    int vram_groups = 0, vram_entries = 0;
    int pcie_groups = 0, pcie_entries = 0;
};

// Small nonnegative cost fits. Counts distinguish a weight read (one expert)
// from FMA work (its tokens). Device stamps measure the GPU span independently
// of the CPU span; a host wait is not mistaken for GPU compute time.
template<int N> struct BatchCostFit {
    double a[N][N] = {}, b[N] = {}, coef[N] = {};
    int samples = 0;
    void sample(const std::array<double, N>& x, double y) {
        if (!std::isfinite(y) || y < 0 || y > 1000) return;
        constexpr double forget = 0.998;
        for (int i = 0; i < N; ++i) {
            b[i] = forget * b[i] + x[i] * y;
            for (int j = 0; j < N; ++j) a[i][j] = forget * a[i][j] + x[i] * x[j];
        }
        ++samples;
    }
    void solve() {
        double best = std::numeric_limits<double>::infinity();
        double winner[N] = {};
        // N <= 5: enumerate active sets, so a correlated/underdetermined fit
        // cannot invent negative compute or memory costs.
        for (int mask = 1; mask < (1 << N); ++mask) {
            int idx[N], n = 0;
            for (int i = 0; i < N; ++i) if (mask & (1 << i)) idx[n++] = i;
            double m[N][N + 1] = {};
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) m[i][j] = a[idx[i]][idx[j]];
                m[i][i] += 1e-5 * std::max(a[idx[i]][idx[i]], 1.0);
                m[i][n] = b[idx[i]];
            }
            bool ok = true;
            for (int k = 0; k < n; ++k) {
                int pivot = k;
                for (int i = k + 1; i < n; ++i) if (std::abs(m[i][k]) > std::abs(m[pivot][k])) pivot = i;
                if (std::abs(m[pivot][k]) < 1e-12) { ok = false; break; }
                for (int j = k; j <= n; ++j) std::swap(m[k][j], m[pivot][j]);
                const double d = m[k][k];
                for (int j = k; j <= n; ++j) m[k][j] /= d;
                for (int i = 0; i < n; ++i) if (i != k) {
                    const double f = m[i][k];
                    for (int j = k; j <= n; ++j) m[i][j] -= f * m[k][j];
                }
            }
            double c[N] = {};
            for (int i = 0; i < n; ++i) {
                c[idx[i]] = m[i][n];
                ok = ok && std::isfinite(c[idx[i]]) && c[idx[i]] >= 0;
            }
            if (!ok) continue;
            double loss = 0;
            for (int i = 0; i < N; ++i) {
                loss -= 2 * c[i] * b[i];
                for (int j = 0; j < N; ++j) loss += c[i] * a[i][j] * c[j];
            }
            if (loss < best) { best = loss; std::copy(c, c + N, winner); }
        }
        std::copy(winner, winner + N, coef);
    }
    double predict(const std::array<double, N>& x) const {
        double y = 0;
        for (int i = 0; i < N; ++i) y += coef[i] * x[i];
        return y;
    }
};

struct UnifiedExpertPlan {
    bool on = false;
    bool paused = false; // an explicit per-request PCIe fraction keeps its requested ownership policy
    struct WidthFit {
        BatchCostFit<4> cpu;
        BatchCostFit<5> gpu;
        int min_pcie = 9999, max_pcie = 0;
        bool ready() const { return cpu.samples >= 64 && gpu.samples >= 64 && max_pcie > min_pcie; }
        double cost(const BatchExpertWork& w) const {
            const double c = cpu.predict({1.0, (double) w.cpu_groups, (double) w.cpu_entries, (double) w.pcie_groups});
            const double g = gpu.predict({1.0, (double) w.vram_groups, (double) w.vram_entries,
                                          (double) w.pcie_groups, (double) w.pcie_entries});
            return std::max(c, g);
        }
    };
    WidthFit widths[8];
    uint64_t decisions = 0, changed = 0, samples = 0;
    void sample(int width, const BatchExpertWork& w, double cpu_ms, double gpu_ms) {
        if (!on || paused || width < 1 || width > 8 || !std::isfinite(cpu_ms) || !std::isfinite(gpu_ms) ||
            cpu_ms < 0 || gpu_ms < 0 || cpu_ms > 1000 || gpu_ms > 1000) return;
        auto& f = widths[width - 1];
        f.cpu.sample({1.0, (double) w.cpu_groups, (double) w.cpu_entries, (double) w.pcie_groups}, cpu_ms);
        f.gpu.sample({1.0, (double) w.vram_groups, (double) w.vram_entries,
                      (double) w.pcie_groups, (double) w.pcie_entries}, gpu_ms);
        f.min_pcie = std::min(f.min_pcie, w.pcie_groups);
        f.max_pcie = std::max(f.max_pcie, w.pcie_groups);
        ++samples;
    }
    void solve(int width) {
        if (width < 1 || width > 8) return;
        auto& f = widths[width - 1];
        if (f.cpu.samples >= 64) { f.cpu.solve(); f.gpu.solve(); }
    }
    // Return one ownership bit per distinct missing expert. All of its token
    // entries stay on that owner, preserving weight reuse and router row order.
    bool choose(int width, const int* nt, int n, int capacity, int vram_groups, int vram_entries,
                bool* selected) {
        if (!on || paused || width < 1 || width > 8 || n < 1 || n > 128 || !widths[width - 1].ready()) return false;
        int total = 0, fixed_n = 0, fixed_t = 0;
        for (int i = 0; i < n; ++i) {
            if (nt[i] < 1 || nt[i] > width) return false;
            total += nt[i];
            if (selected[i]) { ++fixed_n; fixed_t += nt[i]; }
        }
        capacity = std::clamp(capacity, 0, n);
        if (total > 128 || fixed_n > capacity) return false;
        struct Mask { uint64_t lo = 0, hi = 0; };
        static thread_local std::vector<Mask> dp;
        const int stride = total + 1;
        dp.assign((size_t) (capacity + 1) * stride, {});
        int seen_t = 0;
        for (int i = 0; i < n; ++i) {
            seen_t += nt[i];
            for (int q = std::min(capacity, i + 1); q >= 1; --q) for (int t = seen_t; t >= nt[i]; --t) {
                const Mask prev = dp[(size_t) (q - 1) * stride + t - nt[i]];
                if ((q == 1 && t == nt[i]) || prev.lo || prev.hi) {
                    Mask& dst = dp[(size_t) q * stride + t];
                    dst = prev;
                    if (i < 64) dst.lo |= uint64_t(1) << i;
                    else dst.hi |= uint64_t(1) << (i - 64);
                }
            }
        }
        const auto& fit = widths[width - 1];
        auto cost = [&](int q, int t) { return fit.cost({n - q, total - t, vram_groups, vram_entries, q, t}); };
        const double fixed_cost = cost(fixed_n, fixed_t);
        double best = fixed_cost * 0.99; // preserve the existing split for a predicted gain below 1%
        int best_q = -1, best_t = 0;
        for (int q = 0; q <= capacity; ++q) for (int t = 0; t <= total; ++t) {
            const Mask mask = dp[(size_t) q * stride + t];
            if (!(q == 0 && t == 0) && !mask.lo && !mask.hi) continue;
            const double c = cost(q, t);
            if (c < best) { best = c; best_q = q; best_t = t; }
        }
        ++decisions;
        if (best_q < 0) return true;
        const Mask mask = dp[(size_t) best_q * stride + best_t];
        bool different = false;
        for (int i = 0; i < n; ++i) {
            const bool pick = i < 64 ? (mask.lo >> i) & 1 : (mask.hi >> (i - 64)) & 1;
            different = different || selected[i] != pick;
            selected[i] = pick;
        }
        changed += different;
        return true;
    }
};
} // namespace strata::core
