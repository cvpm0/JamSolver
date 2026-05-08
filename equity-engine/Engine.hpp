#pragma once

#include "Cards.hpp"
#include "Evaluator.hpp"
#include "Canonicalise.hpp"

constexpr int    NUM_TRIALS_2WAY = 1'000'000; // 1'000'000 when not testing
constexpr int    NUM_TRIALS_3WAY = 200'000; // 500'000 when not testing
constexpr int    NUM_TRIALS_4WAY = 10'000; // 50'000 when not testing - unused

constexpr double EQUITY_SCALE    = 65535.0;

constexpr uint32_t C(int n, int k) noexcept {
    if (n < k || n < 0) return 0;
    uint32_t result = 1;
    for (int i = 0; i < k; ++i) {
        result *= (n - i);
        result /= (i + 1);
    }
    return result;
}

constexpr uint32_t NUM_2WAY = C(170, 2); // C(170,2) = 14,365
constexpr uint32_t NUM_3WAY = C(171, 3); // C(171,3) = 815,815
constexpr uint32_t NUM_4WAY = C(172, 4); // C(172,4) = ~35M


// combinations with repetition — allows a == b == c == d
// requires sorted inputs a <= b <= c <= d
uint32_t combo_index(int a, int b) noexcept { return C(a, 1) + C(b + 1, 2); }
uint32_t combo_index(int a, int b, int c) noexcept { return C(a, 1) + C(b + 1, 2) + C(c + 2, 3); }
uint32_t combo_index(int a, int b, int c, int d) noexcept { return C(a, 1) + C(b + 1, 2) + C(c + 2, 3) + C(d + 3, 4); }

constexpr auto CLASS_TO_HAND = []() {
    std::array<Hand, 169> table{};
    int idx = 0;

    // pairs: AA down to 22
    for (int r = 12; r >= 0; --r)
        table[idx++] = Hand(make_card(r, 0), make_card(r, 1));

    // suited: AKs down to 32s
    for (int hi = 12; hi >= 1; --hi)
        for (int lo = hi - 1; lo >= 0; --lo)
            table[idx++] = Hand(make_card(hi, 0), make_card(lo, 0));

    // offsuit: AKo down to 32o
    for (int hi = 12; hi >= 1; --hi)
        for (int lo = hi - 1; lo >= 0; --lo)
            table[idx++] = Hand(make_card(hi, 0), make_card(lo, 1));

    return table;
}();

constexpr int is_pair(int hand_class) { return (hand_class <= 12) ? 1 : 0; }

constexpr int is_suited(int hand_class) { return (hand_class >= 13 && hand_class <= 90) ? 1 : 0; }


void run_montecarlo(const Hand* hands, int n, PCG32& rng, Deck& deck,
                    double* equity_out) noexcept {

    deck.reset();

    for (int i = 0; i < n; ++i) {
        deck.remove(hands[i].hi);
        deck.remove(hands[i].lo);
    }

    deck.save();

    double wins[4] = {};

    int trials = (n == 2) ? NUM_TRIALS_2WAY :
             (n == 3) ? NUM_TRIALS_3WAY :
                        NUM_TRIALS_4WAY;

    for (int t = 0; t < trials; ++t) {
        Card board[5];
        deck.draw_board(rng, board, 5);

        HandStrength strengths[4];
        HandStrength best = 0;
        for (int p = 0; p < n; ++p) {
            Card hole[2] = {hands[p].hi, hands[p].lo};
            strengths[p] = evaluate7(hole, board);
            if (strengths[p] > best) best = strengths[p];
        }

        int num_best = 0;
        for (int p = 0; p < n; ++p)
            num_best += (strengths[p] == best);

        double share = 1.0 / num_best;
        for (int p = 0; p < n; ++p)
            if (strengths[p] == best) wins[p] += share;
    }

    for (int p = 0; p < n; ++p) {
        equity_out[p] = wins[p] / trials;
    }

}

void run_matchups_2way(
    int       hero_start,
    int       hero_end,
    std::vector<uint16_t>& table,
    PCG32&    rng,
    Deck&     deck
) noexcept {
    for (int hero = hero_start; hero < hero_end; ++hero) {
        for (int villain = hero; villain < 169; ++villain) {

            uint32_t idx = combo_index(hero, villain);

            // same class — exact 50/50 by suit symmetry, no simulation needed
            if (hero == villain) {
                table[idx * 2 + 0] = (uint16_t)(0.5 * EQUITY_SCALE);
                table[idx * 2 + 1] = (uint16_t)(0.5 * EQUITY_SCALE);
                continue;
            }

            int classes[2] = {hero, villain};
            Hand hands[2];
            if (!canonicalize(classes, 2, hands)) continue;

            double equity[2] = {};
            run_montecarlo(hands, 2, rng, deck, equity);

            table[idx * 2 + 0] = (uint16_t)(equity[0] * EQUITY_SCALE);
            table[idx * 2 + 1] = (uint16_t)(equity[1] * EQUITY_SCALE);
        }
    }
}

void run_matchups_3way(
    int       hero_start,
    int       hero_end,
    std::vector<uint16_t>& table,
    PCG32&    rng,
    Deck&     deck
) noexcept {
    for (int hero = hero_start; hero < hero_end; ++hero) {
        for (int v1 = hero; v1 < 169; ++v1) {
            for (int v2 = v1; v2 < 169; ++v2) {

                uint32_t idx = combo_index(hero, v1, v2);

                int classes[3] = {hero, v1, v2};
                Hand hands[3];
                if (!canonicalize(classes, 3, hands)) continue;

                double equity[3] = {};
                run_montecarlo(hands, 3, rng, deck, equity);

                table[idx * 3 + 0] = (uint16_t)(equity[0] * EQUITY_SCALE);
                table[idx * 3 + 1] = (uint16_t)(equity[1] * EQUITY_SCALE);
                table[idx * 3 + 2] = (uint16_t)(equity[2] * EQUITY_SCALE);
            }
        }
    }
}

void run_matchups_4way(
    int       hero_start,
    int       hero_end,
    std::vector<uint16_t>& table,
    PCG32&    rng,
    Deck&     deck
) noexcept {
    for (int hero = hero_start; hero < hero_end; ++hero) {
        for (int v1 = hero; v1 < 169; ++v1) {
            for (int v2 = v1; v2 < 169; ++v2) {
                for (int v3 = v2; v3 < 169; ++v3) {

                    uint32_t idx = combo_index(hero, v1, v2, v3);

                    int classes[4] = {hero, v1, v2, v3};
                    Hand hands[4];
                    if (!canonicalize(classes, 4, hands)) continue;

                    double equity[4] = {};
                    run_montecarlo(hands, 4, rng, deck, equity);

                    table[idx * 4 + 0] = (uint16_t)(equity[0] * EQUITY_SCALE);
                    table[idx * 4 + 1] = (uint16_t)(equity[1] * EQUITY_SCALE);
                    table[idx * 4 + 2] = (uint16_t)(equity[2] * EQUITY_SCALE);
                    table[idx * 4 + 3] = (uint16_t)(equity[3] * EQUITY_SCALE);
                }
            }
        }
    }
}