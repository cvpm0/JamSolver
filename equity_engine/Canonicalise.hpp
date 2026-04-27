#pragma once

#include "Cards.hpp"

// Per-class concrete hand realizations.
// Layout mirrors CLASS_TO_HAND:
//   0-12:   pairs AA→22        (6 combos, C(4,2))
//   13-90:  suited AKs→32s     (4 combos)
//   91-168: offsuit AKo→32o   (12 combos, 4×3)
// Total table: 169 × 26 bytes ≈ 4.4 KB — L1 resident

struct ClassCombos {
    Hand    combos[12];
    uint8_t count;
};

constexpr std::array<ClassCombos, 169> HAND_COMBOS = []() {
    std::array<ClassCombos, 169> t{};
    int idx = 0;

    // pairs: C(4,2) = 6 suit combinations each
    for (int r = 12; r >= 0; --r) {
        auto& cc = t[idx++];
        cc.count = 0;
        for (int s1 = 0; s1 < 4; ++s1)
            for (int s2 = s1 + 1; s2 < 4; ++s2)
                cc.combos[cc.count++] = Hand(make_card(r, s1), make_card(r, s2));
    }

    // suited: 4 suit choices
    for (int hi = 12; hi >= 1; --hi)
        for (int lo = hi - 1; lo >= 0; --lo) {
            auto& cc = t[idx++];
            cc.count = 0;
            for (int s = 0; s < 4; ++s)
                cc.combos[cc.count++] = Hand(make_card(hi, s), make_card(lo, s));
        }

    // offsuit: 4×3 = 12 suit combinations
    for (int hi = 12; hi >= 1; --hi)
        for (int lo = hi - 1; lo >= 0; --lo) {
            auto& cc = t[idx++];
            cc.count = 0;
            for (int s1 = 0; s1 < 4; ++s1)
                for (int s2 = 0; s2 < 4; ++s2) {
                    if (s1 == s2) continue;
                    cc.combos[cc.count++] = Hand(make_card(hi, s1), make_card(lo, s2));
                }
        }

    return t;
}();

// Bitmask collision check — cards are 0-51, so uint64_t covers all slots.
inline bool card_collision(const Hand* hands, int n) noexcept {
    uint64_t seen = 0;
    for (int i = 0; i < n; ++i) {
        uint64_t bits = (1ULL << hands[i].hi) | (1ULL << hands[i].lo);
        if (seen & bits) return true;
        seen |= bits;
    }
    return false;
}

// Fills hands[] with the lexicographically first non-colliding concrete
// assignment for the given hand classes (may repeat).
// Returns false only if no valid assignment exists (e.g. three hands of
// the same pair class — impossible with 4 suits × 2 cards = 8 aces).
inline bool canonicalize(const int* classes, int n, Hand* hands) noexcept {
    int idx[4] = {};

    for (;;) {
        for (int i = 0; i < n; ++i)
            hands[i] = HAND_COMBOS[classes[i]].combos[idx[i]];

        if (!card_collision(hands, n)) return true;

        // odometer increment from rightmost position
        int carry = 1;
        for (int i = n - 1; i >= 0 && carry; --i) {
            if (++idx[i] < HAND_COMBOS[classes[i]].count)
                carry = 0;
            else
                idx[i] = 0;
        }
        if (carry) return false;
    }
}
