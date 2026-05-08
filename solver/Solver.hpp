#pragma once

#include "Equity.hpp"
#include <algorithm>
#include <cstdint>

// Compile with OpenMP for parallel cache fills:
//   g++ -O3 -fopenmp ...
// If OpenMP is unavailable the #pragma lines are silently ignored and the
// solver still works correctly, just sequentially.

constexpr int    NUM_STATES     = 14;
constexpr int    NUM_HANDS      = 169;
constexpr int    NUM_ITERATIONS = 100'000;
constexpr double STACK          = 8.0;
constexpr double SB_BLIND       = 0.5;
constexpr double BB_BLIND       = 1.0;
constexpr double RAKE           = 0.22;  // flat fee per hand, deducted from pot

enum State : uint8_t {
    UTG    = 0,
    BTN_F  = 1,  BTN_J  = 2,
    SB_FF  = 3,  SB_FJ  = 4,  SB_JF  = 5,  SB_JJ  = 6,
    BB_FFJ = 7,  BB_FJF = 8,  BB_FJJ = 9,
    BB_JFF = 10, BB_JFJ = 11, BB_JJF = 12, BB_JJJ = 13
};

// ── CFR data ─────────────────────────────────────────────────────────────────
inline double strategy[NUM_STATES][NUM_HANDS];      // current jam probability
inline double strategy_sum[NUM_STATES][NUM_HANDS];  // linear-weighted accumulator
inline double regret_jam[NUM_STATES][NUM_HANDS];
inline double regret_fold[NUM_STATES][NUM_HANDS];

// ── per-iteration cache: hero's equity vs villain jam range(s), per hero hand
// eq_vs_one[vs][h]      = hero h's equity vs villain in state vs (HU showdown)
// eq_vs_two[vs1][vs2][h]= hero h's equity vs jamming villains in vs1 and vs2 (3-way)
// Filled once per iteration after strategy update; EV functions just read.
inline double eq_vs_one[NUM_STATES][NUM_HANDS];
inline double eq_vs_two[NUM_STATES][NUM_STATES][NUM_HANDS];
inline double eq_vs_three[NUM_STATES][NUM_STATES][NUM_STATES][NUM_HANDS];

// EV when hero folds, indexed by state (= what hero has already posted, negated)
constexpr double FOLD_EV[NUM_STATES] = {
    0, 0, 0, -0.5, -0.5, -0.5, -0.5, -1, -1, -1, -1, -1, -1, -1
};

// ── combo weighting ──────────────────────────────────────────────────────────
double combo_weight(int h) {
    if (is_pair(h))   return 6.0;
    if (is_suited(h)) return 4.0;
    return 12.0;
}

// ── core idea ────────────────────────────────────────────────────────────────
// For hero's specific hand h in state s, we walk forward through the tree.
// At each future decision node owned by villain v_state, we read p_jam[v_state]
// to weight that villain's jam vs fold branches.
// At showdown nodes, we read eq_vs_one[vs][h] or eq_vs_two[vs1][vs2][h] —
// hero's per-hand equity vs villain(s)' weighted jam range. The cache makes
// this O(1) instead of summing 169^N villain combos per EV call.

// ── declarations ─────────────────────────────────────────────────────────────

// Probability that the player in state s jams (weighted by combo frequency).
double p_jam(State s);

// Cache fillers — called once per iteration after strategy update.
void fill_eq_vs_one();
void fill_eq_vs_two();
void fill_equity_cache();

// EV functions — read from cache, no per-call villain summation.
double ev_jam_bb(State s, int h);
double ev_jam_sb(State s, int h);
double ev_jam_btn(State s, int h);
double ev_jam_utg(int h);
double compute_ev_jam(State s, int h);

// CFR mechanics
void update_strategy(State s, int h);
void update_regrets(State s, int h, int iter);
void run_cfr();
void compute_avg_strategy(double out[NUM_STATES][NUM_HANDS]);

// ═════════════════════════════════════════════════════════════════════════════
// Implementations
// ═════════════════════════════════════════════════════════════════════════════

double p_jam(State s) {
    double total = 0.0, jam = 0.0;
    for (int v = 0; v < NUM_HANDS; ++v) {
        double w = combo_weight(v);
        total += w;
        jam   += w * strategy[s][v];
    }
    return (total > 1e-9) ? jam / total : 0.0;
}

// ── eq_vs_one: fill for all states ──────────────────────────────────────────
// Parallelized across states (independent writes to eq_vs_one[s][...]).
// Inner villain loop iterates only nonzero-weight hands — after convergence
// most states have a small jam range (~5-30 hands), so this is a big win.
void fill_eq_vs_one() {
    #pragma omp parallel for schedule(dynamic)
    for (int s = 0; s < NUM_STATES; ++s) {
        double w_v[NUM_HANDS];
        int    nz_idx[NUM_HANDS];
        int    nz_count = 0;
        double denom = 0.0;

        for (int v = 0; v < NUM_HANDS; ++v) {
            double w = combo_weight(v) * strategy[s][v];
            w_v[v] = w;
            if (w > 0.0) {
                nz_idx[nz_count++] = v;
                denom += w;
            }
        }
        if (denom < 1e-9) {
            for (int h = 0; h < NUM_HANDS; ++h) eq_vs_one[s][h] = 0.5;
            continue;
        }
        double inv_denom = 1.0 / denom;
        for (int h = 0; h < NUM_HANDS; ++h) {
            double num = 0.0;
            for (int i = 0; i < nz_count; ++i) {
                int v = nz_idx[i];
                num += w_v[v] * get_equity_2way(h, v, 0);
            }
            eq_vs_one[s][h] = num * inv_denom;
        }
    }
}

// Helper: fill eq_vs_two[vs1][vs2] for one specific pair.
// After building villain weights, we compact to lists of nonzero indices.
// At convergence, ranges are sparse (~5-30 hands), so iterating compacted
// lists is dramatically faster than skipping zeros inline (30-100x speedup
// in late iterations).
static void fill_eq_vs_two_pair(State vs1, State vs2) {
    double w1[NUM_HANDS], w2[NUM_HANDS];
    int    idx1[NUM_HANDS], idx2[NUM_HANDS];
    int    n1 = 0, n2 = 0;
    double sum1 = 0.0, sum2 = 0.0;

    for (int v = 0; v < NUM_HANDS; ++v) {
        double a = combo_weight(v) * strategy[vs1][v];
        double b = combo_weight(v) * strategy[vs2][v];
        w1[v] = a; w2[v] = b;
        if (a > 0.0) { idx1[n1++] = v; sum1 += a; }
        if (b > 0.0) { idx2[n2++] = v; sum2 += b; }
    }
    double denom = sum1 * sum2;
    if (denom < 1e-9) {
        for (int h = 0; h < NUM_HANDS; ++h) eq_vs_two[vs1][vs2][h] = 1.0/3.0;
        return;
    }
    double inv_denom = 1.0 / denom;

    for (int h = 0; h < NUM_HANDS; ++h) {
        double num = 0.0;
        for (int i = 0; i < n1; ++i) {
            int    v1 = idx1[i];
            double wa = w1[v1];
            for (int j = 0; j < n2; ++j) {
                int v2 = idx2[j];
                num += wa * w2[v2] * get_equity_3way(h, v1, v2, 0);
            }
        }
        eq_vs_two[vs1][vs2][h] = num * inv_denom;
    }
}

// ── eq_vs_two: only fill the (vs1, vs2) pairs that EV functions actually use.
// Parallelized across pairs — each pair writes to a disjoint slice of
// eq_vs_two[][][], so no synchronization needed.
void fill_eq_vs_two() {
    // Pairs referenced by ev_jam_* functions (including 4-way approximations)
    static constexpr int PAIRS[][2] = {
        {BTN_F,  SB_FJ},     // BB_FJJ; BTN_F SB-call/BB-call
        {UTG,    SB_JF},     // BB_JFJ; UTG BTN-fold/SB-call/BB-call
        {UTG,    BTN_J},     // BB_JJF; 4-way approx for BB_JJJ and SB_JJ all-call
        {BTN_F,  BB_FJJ},    // SB_FJ BB-call
        {UTG,    BB_JFJ},    // SB_JF BB-call
        {SB_FJ,  BB_FJJ},    // BTN_F SB-call/BB-call
        {UTG,    BB_JJF},    // BTN_J SB-fold/BB-call; UTG BTN-call/SB-fold/BB-call
        {BTN_J,  SB_JJ},     // UTG BTN-call/SB-call/BB-fold; 4-way approx for UTG all-call
        {BTN_J,  BB_JJF},    // UTG BTN-call/SB-fold/BB-call
        {SB_JF,  BB_JFJ},    // UTG BTN-fold/SB-call/BB-call
        {UTG,    SB_JJ},     // 4-way approx for BTN_J all-call
    };
    constexpr int N_PAIRS = sizeof(PAIRS) / sizeof(PAIRS[0]);

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < N_PAIRS; ++i) {
        fill_eq_vs_two_pair(static_cast<State>(PAIRS[i][0]),
                            static_cast<State>(PAIRS[i][1]));
    }
}

// Helper: fill eq_vs_three[vs1][vs2][vs3] for one specific triple.
// Triple-nested compacted loop: n1 × n2 × n3 × 169 hero hands.
// After convergence, n1/n2/n3 are typically 5-30, so the inner work
// is roughly 25³ × 169 ≈ 2.6M ops per triple — fast.
static void fill_eq_vs_three_triple(State vs1, State vs2, State vs3) {
    double w1[NUM_HANDS], w2[NUM_HANDS], w3[NUM_HANDS];
    int    idx1[NUM_HANDS], idx2[NUM_HANDS], idx3[NUM_HANDS];
    int    n1 = 0, n2 = 0, n3 = 0;
    double sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;

    for (int v = 0; v < NUM_HANDS; ++v) {
        double a = combo_weight(v) * strategy[vs1][v];
        double b = combo_weight(v) * strategy[vs2][v];
        double c = combo_weight(v) * strategy[vs3][v];
        w1[v] = a; w2[v] = b; w3[v] = c;
        if (a > 0.0) { idx1[n1++] = v; sum1 += a; }
        if (b > 0.0) { idx2[n2++] = v; sum2 += b; }
        if (c > 0.0) { idx3[n3++] = v; sum3 += c; }
    }
    double denom = sum1 * sum2 * sum3;
    if (denom < 1e-9) {
        for (int h = 0; h < NUM_HANDS; ++h) eq_vs_three[vs1][vs2][vs3][h] = 0.25;
        return;
    }
    double inv_denom = 1.0 / denom;

    for (int h = 0; h < NUM_HANDS; ++h) {
        double num = 0.0;
        for (int i = 0; i < n1; ++i) {
            int    v1 = idx1[i];
            double wa = w1[v1];
            for (int j = 0; j < n2; ++j) {
                int    v2 = idx2[j];
                double wb = wa * w2[v2];
                for (int k = 0; k < n3; ++k) {
                    int v3 = idx3[k];
                    num += wb * w3[v3] * get_equity_4way(h, v1, v2, v3, 0);
                }
            }
        }
        eq_vs_three[vs1][vs2][vs3][h] = num * inv_denom;
    }
}

// ── eq_vs_three: only fill the 4 triples that EV functions actually use.
// Parallelised across triples — each writes to a disjoint slice.
void fill_eq_vs_three() {
    static constexpr int TRIPLES[][3] = {
        {UTG,   BTN_J,  SB_JJ},   // BB_JJJ: hero vs all three jammers
        {UTG,   BTN_J,  BB_JJJ},  // SB_JJ BB-calls: hero vs UTG+BTN+BB
        {UTG,   SB_JJ,  BB_JJJ},  // BTN_J SB+BB call: hero vs UTG+SB+BB
        {BTN_J, SB_JJ,  BB_JJJ},  // UTG all call: hero vs BTN+SB+BB
    };
    constexpr int N_TRIPLES = sizeof(TRIPLES) / sizeof(TRIPLES[0]);

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < N_TRIPLES; ++i) {
        fill_eq_vs_three_triple(static_cast<State>(TRIPLES[i][0]),
                                static_cast<State>(TRIPLES[i][1]),
                                static_cast<State>(TRIPLES[i][2]));
    }
}

void fill_equity_cache() {
    fill_eq_vs_one();
    fill_eq_vs_two();
    fill_eq_vs_three();
}

// ── BB nodes: action closes after BB's decision ─────────────────────────────
double ev_jam_bb(State s, int h) {
    switch (s) {
        case BB_FFJ:
            return eq_vs_one[SB_FF][h] * (16.0 - RAKE) - STACK;

        case BB_FJF:
            return eq_vs_one[BTN_F][h] * (16.5 - RAKE) - STACK;

        case BB_FJJ:
            return eq_vs_two[BTN_F][SB_FJ][h] * (24.0 - RAKE) - STACK;

        case BB_JFF:
            return eq_vs_one[UTG][h] * (16.5 - RAKE) - STACK;

        case BB_JFJ:
            return eq_vs_two[UTG][SB_JF][h] * (24.0 - RAKE) - STACK;

        case BB_JJF:
            return eq_vs_two[UTG][BTN_J][h] * (24.5 - RAKE) - STACK;

        case BB_JJJ:
            return eq_vs_three[UTG][BTN_J][SB_JJ][h] * (32.0 - RAKE) - STACK;

        default:
            return 0.0;
    }
}

// ── SB nodes: BB still to act ───────────────────────────────────────────────
double ev_jam_sb(State s, int h) {
    switch (s) {
        case SB_FF: {
            double pc = p_jam(BB_FFJ);
            double pf = 1.0 - pc;
            return pf * ((9.0 - RAKE) - STACK)
                 + pc * (eq_vs_one[BB_FFJ][h] * (16.0 - RAKE) - STACK);
        }

        case SB_FJ: {
            double pc = p_jam(BB_FJJ);
            double pf = 1.0 - pc;
            return pf * (eq_vs_one[BTN_F][h] * (17.0 - RAKE) - STACK)
                 + pc * (eq_vs_two[BTN_F][BB_FJJ][h] * (24.0 - RAKE) - STACK);
        }

        case SB_JF: {
            double pc = p_jam(BB_JFJ);
            double pf = 1.0 - pc;
            return pf * (eq_vs_one[UTG][h] * (17.0 - RAKE) - STACK)
                 + pc * (eq_vs_two[UTG][BB_JFJ][h] * (24.0 - RAKE) - STACK);
        }

        case SB_JJ: {
            double pc = p_jam(BB_JJJ);
            double pf = 1.0 - pc;
            return pf * (eq_vs_two[UTG][BTN_J][h] * (25.0 - RAKE) - STACK)
                 + pc * (eq_vs_three[UTG][BTN_J][BB_JJJ][h] * (32.0 - RAKE) - STACK);
        }

        default:
            return 0.0;
    }
}

// ── BTN nodes: SB and BB still to act ───────────────────────────────────────
double ev_jam_btn(State s, int h) {
    switch (s) {
        case BTN_F: {
            double ps  = p_jam(SB_FJ);
            double pfs = 1.0 - ps;
            double pb_no_sb = p_jam(BB_FJF);
            double pb_sb    = p_jam(BB_FJJ);

            double ev = 0.0;
            ev += pfs * (1.0 - pb_no_sb) * ((9.5 - RAKE) - STACK);
            ev += pfs * pb_no_sb         * (eq_vs_one[BB_FJF][h] * (16.5 - RAKE) - STACK);
            ev += ps  * (1.0 - pb_sb)    * (eq_vs_one[SB_FJ][h]  * (17.0 - RAKE) - STACK);
            ev += ps  * pb_sb            * (eq_vs_two[SB_FJ][BB_FJJ][h] * (24.0 - RAKE) - STACK);
            return ev;
        }

        case BTN_J: {
            double ps  = p_jam(SB_JJ);
            double pfs = 1.0 - ps;
            double pb_no_sb = p_jam(BB_JJF);
            double pb_sb    = p_jam(BB_JJJ);

            double ev = 0.0;
            ev += pfs * (1.0 - pb_no_sb) * (eq_vs_one[UTG][h] * (17.5 - RAKE) - STACK);
            ev += pfs * pb_no_sb         * (eq_vs_two[UTG][BB_JJF][h] * (24.5 - RAKE) - STACK);
            ev += ps  * (1.0 - pb_sb)    * (eq_vs_two[UTG][SB_JJ][h]  * (25.0 - RAKE) - STACK);
            ev += ps  * pb_sb            * (eq_vs_three[UTG][SB_JJ][BB_JJJ][h] * (32.0 - RAKE) - STACK);
            return ev;
        }

        default:
            return 0.0;
    }
}

// ── UTG node: BTN, SB, BB still to act ──────────────────────────────────────
double ev_jam_utg(int h) {
    double pb        = p_jam(BTN_J);
    double pfb       = 1.0 - pb;
    double ps_no_btn = p_jam(SB_JF);
    double ps_btn    = p_jam(SB_JJ);
    double pbb_ff    = p_jam(BB_JFF);
    double pbb_fc    = p_jam(BB_JFJ);
    double pbb_cf    = p_jam(BB_JJF);
    double pbb_cc    = p_jam(BB_JJJ);

    double ev = 0.0;

    ev += pfb * (1.0 - ps_no_btn) * (1.0 - pbb_ff) * ((9.5 - RAKE) - STACK);

    ev += pfb * (1.0 - ps_no_btn) * pbb_ff
        * (eq_vs_one[BB_JFF][h] * (16.5 - RAKE) - STACK);

    ev += pfb * ps_no_btn * (1.0 - pbb_fc)
        * (eq_vs_one[SB_JF][h] * (17.0 - RAKE) - STACK);

    ev += pfb * ps_no_btn * pbb_fc
        * (eq_vs_two[SB_JF][BB_JFJ][h] * (24.0 - RAKE) - STACK);

    ev += pb * (1.0 - ps_btn) * (1.0 - pbb_cf)
        * (eq_vs_one[BTN_J][h] * (17.5 - RAKE) - STACK);

    ev += pb * (1.0 - ps_btn) * pbb_cf
        * (eq_vs_two[BTN_J][BB_JJF][h] * (24.5 - RAKE) - STACK);

    ev += pb * ps_btn * (1.0 - pbb_cc)
        * (eq_vs_two[BTN_J][SB_JJ][h] * (25.0 - RAKE) - STACK);

    ev += pb * ps_btn * pbb_cc
        * (eq_vs_three[BTN_J][SB_JJ][BB_JJJ][h] * (32.0 - RAKE) - STACK);

    return ev;
}

double compute_ev_jam(State s, int h) {
    if (s >= BB_FFJ) return ev_jam_bb(s, h);
    if (s >= SB_FF)  return ev_jam_sb(s, h);
    if (s >= BTN_F)  return ev_jam_btn(s, h);
    return ev_jam_utg(h);
}

// ── strategy / regret updates ───────────────────────────────────────────────
void update_strategy(State s, int h) {
    double rj = std::max(0.0, regret_jam[s][h]);
    double rf = std::max(0.0, regret_fold[s][h]);
    double total = rj + rf;
    strategy[s][h] = (total > 0.0) ? rj / total : 0.5;
}

void update_regrets(State s, int h, int iter) {
    double ev_j = compute_ev_jam(s, h);
    double ev_f = FOLD_EV[s];

    double sigma   = strategy[s][h];
    double node_ev = sigma * ev_j + (1.0 - sigma) * ev_f;

    // CFR+: floor at zero immediately
    regret_jam[s][h]  = std::max(0.0, regret_jam[s][h]  + ev_j - node_ev);
    regret_fold[s][h] = std::max(0.0, regret_fold[s][h] + ev_f - node_ev);

    // Linear weighting: iteration t contributes proportional to t
    strategy_sum[s][h] += iter * sigma;
}

void run_cfr() {
    for (int s = 0; s < NUM_STATES; ++s)
        for (int h = 0; h < NUM_HANDS; ++h)
            strategy[s][h] = 0.5;

    for (int i = 1; i <= NUM_ITERATIONS; ++i) {
        // 1) update strategy from accumulated regrets
        for (int s = 0; s < NUM_STATES; ++s)
            for (int h = 0; h < NUM_HANDS; ++h)
                update_strategy(static_cast<State>(s), h);

        // 2) rebuild equity cache from new strategy (the expensive step)
        fill_equity_cache();

        // 3) update regrets using cached equities (fast)
        for (int s = 0; s < NUM_STATES; ++s)
            for (int h = 0; h < NUM_HANDS; ++h)
                update_regrets(static_cast<State>(s), h, i);
    }
}

void compute_avg_strategy(double out[NUM_STATES][NUM_HANDS]) {
    // Triangular number: 1 + 2 + ... + NUM_ITERATIONS.
    // Promote to int64 — at NUM_ITERATIONS = 100k this overflows int32.
    int64_t n = static_cast<int64_t>(NUM_ITERATIONS);
    double total_weight = static_cast<double>(n * (n + 1)) / 2.0;
    for (int s = 0; s < NUM_STATES; ++s)
        for (int h = 0; h < NUM_HANDS; ++h)
            out[s][h] = (total_weight > 0.0) ? strategy_sum[s][h] / total_weight : 0.0;
}