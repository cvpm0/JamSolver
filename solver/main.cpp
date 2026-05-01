#include "Solver.hpp"
#include "../equity_engine/Cards.hpp"

#include <iostream>
#include <iomanip>
#include <string>
#include <array>
#include <cstdio>
#include <cstdint>

// ─────────────────────────────────────────────────────────────────────────────
// Preflop chart printing
// ─────────────────────────────────────────────────────────────────────────────

// Rank order for chart: A K Q J T 9 8 7 6 5 4 3 2 (high → low)
constexpr std::array<char, 13> RANK_LABELS = {
    'A','K','Q','J','T','9','8','7','6','5','4','3','2'
};

// Cards.hpp uses card_rank(c): 0=2, 1=3, ..., 12=A.
// Our chart uses RANK_LABELS in descending order (idx 0=A, idx 12=2).
// Convert: chart_idx = 12 - card_rank.
inline int card_to_chart_idx(Card c) {
    return 12 - card_rank(c);
}

// Build a 13x13 lookup: chart_lookup[row][col] = hand class index in [0, NUM_HANDS).
// Convention: row==col → pair (diagonal)
//             row<col  → suited   (upper-right; row=higher rank, col=lower rank)
//             row>col  → offsuit  (lower-left;  row=lower rank,  col=higher rank)
// We populate it by inspecting every hand class once, deriving suitedness
// directly from the two cards in CLASS_TO_HAND (no reliance on is_suited).
void build_chart_lookup(int chart_lookup[13][13]) {
    for (int r = 0; r < 13; ++r)
        for (int c = 0; c < 13; ++c)
            chart_lookup[r][c] = -1;

    for (int h = 0; h < NUM_HANDS; ++h) {
        Card hi_card = CLASS_TO_HAND[h].hi;  // higher-ranked card per Hand invariant
        Card lo_card = CLASS_TO_HAND[h].lo;
        int  hi = card_to_chart_idx(hi_card);  // smaller chart-idx (higher rank)
        int  lo = card_to_chart_idx(lo_card);  // larger chart-idx (lower rank)

        bool pair   = (card_rank(hi_card) == card_rank(lo_card));
        bool suited = !pair && (card_suit(hi_card) == card_suit(lo_card));

        if (pair) {
            chart_lookup[hi][lo] = h;        // diagonal (hi == lo here)
        } else if (suited) {
            chart_lookup[hi][lo] = h;        // upper-right: row=high, col=low
        } else {
            chart_lookup[lo][hi] = h;        // lower-left:  row=low,  col=high
        }
    }
}

// Format a percentage as a 3-character cell, e.g. "100", " 87", "  4", "  ."
std::string fmt_cell(double pct) {
    if (pct < 0.5) return "  .";   // visually de-emphasise pure folds
    char buf[8];
    std::snprintf(buf, sizeof(buf), "%3d", static_cast<int>(pct + 0.5));
    return std::string(buf);
}

// Human-readable label for each state
const char* state_label(int s) {
    switch (s) {
        case UTG:    return "UTG (open jam)";
        case BTN_F:  return "BTN vs UTG fold";
        case BTN_J:  return "BTN vs UTG jam";
        case SB_FF:  return "SB vs fold-fold (open jam)";
        case SB_FJ:  return "SB vs BTN jam";
        case SB_JF:  return "SB vs UTG jam (BTN folded)";
        case SB_JJ:  return "SB vs UTG+BTN jam";
        case BB_FFJ: return "BB vs SB jam";
        case BB_FJF: return "BB vs BTN jam (SB folded)";
        case BB_FJJ: return "BB vs BTN jam, SB called";
        case BB_JFF: return "BB vs UTG jam (BTN+SB folded)";
        case BB_JFJ: return "BB vs UTG jam, SB called";
        case BB_JJF: return "BB vs UTG+BTN jam (SB folded)";
        case BB_JJJ: return "BB vs UTG+BTN+SB jam";
        default: return "?";
    }
}

void print_chart(int s, double avg[NUM_STATES][NUM_HANDS],
                 const int chart_lookup[13][13])
{
    std::cout << "\n═══════════════════════════════════════════════════════════\n";
    std::cout << "  State " << s << ": " << state_label(s) << "\n";
    std::cout << "  (suited = upper-right, offsuit = lower-left, pairs on diagonal)\n";
    std::cout << "═══════════════════════════════════════════════════════════\n";

    // header row
    std::cout << "    ";
    for (int c = 0; c < 13; ++c) std::cout << "  " << RANK_LABELS[c] << " ";
    std::cout << "\n";

    for (int r = 0; r < 13; ++r) {
        std::cout << "  " << RANK_LABELS[r] << " ";
        for (int c = 0; c < 13; ++c) {
            int h = chart_lookup[r][c];
            if (h < 0) {
                std::cout << "  ? ";
            } else {
                std::cout << " " << fmt_cell(avg[s][h] * 100.0) << " ";
            }
        }
        std::cout << "\n";
    }

    // jam frequency summary
    double total_w = 0.0, jam_w = 0.0;
    for (int h = 0; h < NUM_HANDS; ++h) {
        double w = combo_weight(h);
        total_w += w;
        jam_w   += w * avg[s][h];
    }
    double pct = (total_w > 0.0) ? 100.0 * jam_w / total_w : 0.0;
    std::cout << "\n  Overall jam frequency: " << std::fixed
              << std::setprecision(1) << pct << "%\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Binary serialization
// ─────────────────────────────────────────────────────────────────────────────
//
// File format (little-endian, packed):
//   bytes 0..3   : magic   = 0x4A414D53  ("JAMS")
//   bytes 4..7   : version = 1
//   bytes 8..11  : num_states (uint32) — should be 14
//   bytes 12..15 : num_hands  (uint32) — should be 169
//   bytes 16..   : double[num_states][num_hands] in row-major order
//                  (state-major: state 0 hands first, then state 1, ...)
//
// To read a single (state, hand) jam probability:
//   offset = 16 + (state * num_hands + hand) * sizeof(double)
//   fseek(f, offset, SEEK_SET); fread(&p, sizeof(double), 1, f);
//
// Total size = 16 + 14*169*8 = 18,944 bytes.

constexpr uint32_t JAMS_MAGIC   = 0x4A414D53;  // "JAMS"
constexpr uint32_t JAMS_VERSION = 1;

bool save_strategy_binary(const char* path,
                          const double avg[NUM_STATES][NUM_HANDS])
{
    FILE* f = std::fopen(path, "wb");
    if (!f) {
        std::cerr << "Failed to open " << path << " for writing\n";
        return false;
    }

    uint32_t header[4] = {
        JAMS_MAGIC,
        JAMS_VERSION,
        static_cast<uint32_t>(NUM_STATES),
        static_cast<uint32_t>(NUM_HANDS)
    };
    if (std::fwrite(header, sizeof(uint32_t), 4, f) != 4) {
        std::cerr << "Header write failed\n"; std::fclose(f); return false;
    }

    size_t expected = NUM_STATES * NUM_HANDS;
    if (std::fwrite(avg, sizeof(double), expected, f) != expected) {
        std::cerr << "Strategy write failed\n"; std::fclose(f); return false;
    }

    std::fclose(f);
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────

int main() {
    load_equity();

    // sanity check: AA vs KK should be ~0.82
    printf("Sanity: AA vs KK = %.3f (expect ~0.82)\n\n", get_equity_2way(0, 1, 0));

    std::cout << "Running CFR for " << NUM_ITERATIONS << " iterations...\n";
    run_cfr();

    double avg_strategy[NUM_STATES][NUM_HANDS];
    compute_avg_strategy(avg_strategy);

    int chart_lookup[13][13];
    build_chart_lookup(chart_lookup);

    for (int s = 0; s < NUM_STATES; ++s) {
        print_chart(s, avg_strategy, chart_lookup);
    }

    // Persist to binary for the retrieval UI.
    const char* out_path = "strategy.jams";
    if (save_strategy_binary(out_path, avg_strategy)) {
        std::cout << "\nStrategy saved to " << out_path
                  << " (" << (16 + NUM_STATES * NUM_HANDS * sizeof(double))
                  << " bytes)\n";
    }

    std::cout << "\nDone.\n";
    return 0;
}