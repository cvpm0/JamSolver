#pragma once

#include "Cards.hpp"

// hand strength encoded as uint32_t
// (category << 20) | (r1 << 16) | (r2 << 12) | (r3 << 8) | (r4 << 4) | r5
// higher value = stronger hand, direct integer comparison
using HandStrength = uint32_t;

// hand category constants
constexpr uint32_t HIGH_CARD       = 0;  // top 5 ranks
constexpr uint32_t PAIR            = 1;  // pair rank + 3 kickers
constexpr uint32_t TWO_PAIR        = 2;  // high pair + low pair + kicker
constexpr uint32_t TRIPS           = 3;  // trip rank + 2 kickers
constexpr uint32_t STRAIGHT        = 4;  // straight_high only
constexpr uint32_t FLUSH           = 5;  // top 5 ranks in flush suit
constexpr uint32_t FULL_HOUSE      = 6;  // trip rank + pair rank
constexpr uint32_t QUADS           = 7;  // quad rank + kicker
constexpr uint32_t STRAIGHT_FLUSH  = 8;  // sf_high only

// Returns the highest-card rank of the best straight in mask, or -1.
// Uses the shift-AND trick: bit h set in the result means ranks h..h+4 are all present.
static inline int highest_straight(uint16_t mask) noexcept {
    uint16_t c = mask & (mask >> 1) & (mask >> 2) & (mask >> 3) & (mask >> 4);
    if (c) return (31 - __builtin_clz((uint32_t)c)) + 4;
    if ((mask & 0x100F) == 0x100F) return 3;  // wheel: A-2-3-4-5
    return -1;
}

[[nodiscard]] HandStrength evaluate7(const Card* hole,
                                     const Card* board) noexcept {

    uint16_t suit_rank_mask[4] = {};
    uint16_t rank_mask  = 0;
    uint16_t rank_2plus = 0;   // ranks with ≥2 cards
    uint16_t rank_3plus = 0;   // ranks with ≥3 cards
    uint16_t rank_4     = 0;   // ranks with 4 cards

    // Single collection pass — no rank_count array, no suit_count array.
    // Multiplicity tracked by bitmask state transitions.
    const Card all7[7] = {hole[0], hole[1], board[0], board[1], board[2], board[3], board[4]};
    for (int i = 0; i < 7; ++i) {
        Card     c   = all7[i];
        uint16_t bit = (uint16_t)(1u << (c >> 2));
        suit_rank_mask[c & 3] |= bit;
        if (rank_mask & bit) {
            if (rank_2plus & bit) {
                if (rank_3plus & bit) rank_4     |= bit;
                else                  rank_3plus |= bit;
            } else {
                rank_2plus |= bit;
            }
        }
        rank_mask |= bit;
    }

    // Flush via popcount — no suit_count array.
    int flush_suit = -1;
    for (int s = 0; s < 4; ++s) {
        if (__builtin_popcount(suit_rank_mask[s]) >= 5) { flush_suit = s; break; }
    }

    // Straight flush
    if (flush_suit != -1) {
        int sf_high = highest_straight(suit_rank_mask[flush_suit]);
        if (sf_high != -1)
            return (STRAIGHT_FLUSH << 20) | ((uint32_t)sf_high << 16);
    }

    // Rank groupings — no 13-iteration scan, all O(1) bit ops.
    uint16_t trips_mask = rank_3plus & ~rank_4;
    uint16_t pairs_mask = rank_2plus & ~rank_3plus;

    int quad_rank = rank_4     ? (31 - __builtin_clz((uint32_t)rank_4))     : -1;
    int trip_rank = trips_mask ? (31 - __builtin_clz((uint32_t)trips_mask)) : -1;

    // Pair candidates: explicit pairs + any second trip acting as pair.
    uint16_t pair_src  = pairs_mask | (trip_rank != -1 ? trips_mask & ~(1u << trip_rank) : 0u);
    int pair_rank      = pair_src  ? (31 - __builtin_clz((uint32_t)pair_src))  : -1;
    uint16_t pair_src2 = pair_rank != -1 ? pair_src & ~(1u << pair_rank) : 0u;
    int pair_rank2     = pair_src2 ? (31 - __builtin_clz((uint32_t)pair_src2)) : -1;

    // Quads — 1 kicker
    if (quad_rank != -1) {
        uint16_t km = rank_mask & ~(1u << quad_rank);
        int kicker = 31 - __builtin_clz((uint32_t)km);
        return (QUADS << 20) | ((uint32_t)quad_rank << 16) | ((uint32_t)kicker << 12);
    }

    // Full house
    if (trip_rank != -1 && pair_rank != -1)
        return (FULL_HOUSE << 20) | ((uint32_t)trip_rank << 16) | ((uint32_t)pair_rank << 12);

    // Flush — top 5 ranks, exactly 5 iterations, no skips
    if (flush_suit != -1) {
        uint16_t fm = suit_rank_mask[flush_suit];
        HandStrength s = FLUSH << 20;
        for (int i = 0; i < 5; ++i) {
            int r = 31 - __builtin_clz((uint32_t)fm);
            s |= (uint32_t)r << (16 - i * 4);
            fm ^= (uint16_t)(1u << r);
        }
        return s;
    }

    // Straight
    int straight_high = highest_straight(rank_mask);
    if (straight_high != -1)
        return (STRAIGHT << 20) | ((uint32_t)straight_high << 16);

    // Trips — 2 kickers, exactly 2 iterations
    if (trip_rank != -1) {
        uint16_t km = rank_mask & ~(1u << trip_rank);
        HandStrength s = (TRIPS << 20) | ((uint32_t)trip_rank << 16);
        for (int i = 0; i < 2; ++i) {
            int r = 31 - __builtin_clz((uint32_t)km);
            s |= (uint32_t)r << (12 - i * 4);
            km ^= (uint16_t)(1u << r);
        }
        return s;
    }

    // Two pair — 1 kicker
    if (pair_rank2 != -1) {
        uint16_t km = rank_mask & ~(1u << pair_rank) & ~(1u << pair_rank2);
        int kicker = 31 - __builtin_clz((uint32_t)km);
        return (TWO_PAIR << 20) | ((uint32_t)pair_rank << 16) | ((uint32_t)pair_rank2 << 12)
                                | ((uint32_t)kicker << 8);
    }

    // Pair — 3 kickers, exactly 3 iterations
    if (pair_rank != -1) {
        uint16_t km = rank_mask & ~(1u << pair_rank);
        HandStrength s = (PAIR << 20) | ((uint32_t)pair_rank << 16);
        for (int i = 0; i < 3; ++i) {
            int r = 31 - __builtin_clz((uint32_t)km);
            s |= (uint32_t)r << (12 - i * 4);
            km ^= (uint16_t)(1u << r);
        }
        return s;
    }

    // High card — top 5 ranks, exactly 5 iterations
    {
        uint16_t km = rank_mask;
        HandStrength s = HIGH_CARD << 20;
        for (int i = 0; i < 5; ++i) {
            int r = 31 - __builtin_clz((uint32_t)km);
            s |= (uint32_t)r << (16 - i * 4);
            km ^= (uint16_t)(1u << r);
        }
        return s;
    }
}
