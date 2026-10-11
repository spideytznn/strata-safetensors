#include "strata/core/unified_expert_plan.hpp"
#include <cstdio>
#include <random>
#include <stdexcept>

using namespace strata::core;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
int main() {
    try {
        BatchCostFit<4> fit;
        std::mt19937 rng(423721);
        const double truth[4] = {.012, .025, .008, .004};
        for (int i = 0; i < 1000; ++i) {
            const std::array<double, 4> x = {1, (double) (rng() % 12), (double) (rng() % 30), (double) (rng() % 8)};
            double y = 0;
            for (int j = 0; j < 4; ++j) y += truth[j] * x[j];
            fit.sample(x, y);
        }
        fit.solve();
        for (int j = 0; j < 4; ++j) require(std::abs(fit.coef[j] - truth[j]) < 1e-5, "cost fit recovery");
        const int before = fit.samples;
        fit.sample({1, 1, 1, 1}, std::numeric_limits<double>::quiet_NaN());
        require(before == fit.samples, "invalid sample accepted");
        UnifiedExpertPlan plan;
        plan.on = true;
        int checks = 0;
        for (int repeat = 0; repeat < 600; ++repeat) {
            const int width = 1 + rng() % 8, n = 1 + rng() % 10, cap = rng() % (n + 1);
            auto& f = plan.widths[width - 1];
            f.cpu.samples = f.gpu.samples = 100;
            f.min_pcie = 0; f.max_pcie = 2;
            for (double& x : f.cpu.coef) x = .001 * (1 + rng() % 40);
            for (double& x : f.gpu.coef) x = .001 * (1 + rng() % 40);
            int nt[128], total = 0, fixed_n = 0, fixed_t = 0;
            bool selected[128] = {};
            for (int i = 0; i < n; ++i) {
                nt[i] = 1 + rng() % width;
                total += nt[i];
                selected[i] = i < cap / 2;
                if (selected[i]) { ++fixed_n; fixed_t += nt[i]; }
            }
            auto cost = [&](int q, int t) { return f.cost({n - q, total - t, 4, 7, q, t}); };
            const double fixed = cost(fixed_n, fixed_t);
            double best = fixed;
            for (int mask = 0; mask < (1 << n); ++mask) {
                int q = 0, t = 0;
                for (int i = 0; i < n; ++i) if (mask & (1 << i)) { ++q; t += nt[i]; }
                if (q <= cap) best = std::min(best, cost(q, t));
            }
            require(plan.choose(width, nt, n, cap, 4, 7, selected), "trained planner did not choose");
            int q = 0, t = 0;
            for (int i = 0; i < n; ++i) if (selected[i]) { ++q; t += nt[i]; }
            require(q <= cap, "staging capacity exceeded");
            require(std::abs(cost(q, t) - (best < fixed * .99 ? best : fixed)) < 1e-10, "subset is not optimal");
            ++checks;
        }
        // Exercise the high mask word and full 128-entry boundary.
        int nt[128]; bool chosen[128] = {};
        std::fill(nt, nt + 128, 1);
        auto& f = plan.widths[0];
        f.cpu.coef[0] = .01; f.cpu.coef[1] = 1; f.cpu.coef[2] = 0; f.cpu.coef[3] = 0;
        std::fill(f.gpu.coef, f.gpu.coef + 5, 0.0);
        require(plan.choose(1, nt, 128, 100, 0, 0, chosen), "128 entries rejected");
        require(std::count(chosen, chosen + 128, true) == 100, "high mask lost entries");
        plan.paused = true;
        require(!plan.choose(1, nt, 128, 100, 0, 0, chosen), "explicit request override was ignored");
        plan.paused = false;
        plan.on = false;
        require(!plan.choose(1, nt, 128, 100, 0, 0, chosen), "disabled planner changed ownership");
        std::printf("unified expert plan: cost recovery, %d exhaustive subset comparisons, bounds passed\n", checks);
    } catch (const std::exception& error) { std::fprintf(stderr, "%s\n", error.what()); return 1; }
    return 0;
}
