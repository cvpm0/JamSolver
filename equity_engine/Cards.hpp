#pragma once

#include <cstdint>
#include <array>
#include <cstring>
#include <iostream>
#include "Random.hpp"

using Card = uint8_t;

// =============================================================================
// CARD ENCODING
// card = rank * 4 + suit
// rank: 0=2, 1=3, 2=4, 3=5, 4=6, 5=7, 6=8, 7=9, 8=T, 9=J, 10=Q, 11=K, 12=A
// suit: 0=c, 1=d, 2=h, 3=s
// =============================================================================

constexpr int  card_rank(Card c)            noexcept { return c >> 2; }
constexpr int  card_suit(Card c)            noexcept { return c & 3; }
constexpr Card make_card(int rank, int suit) noexcept { 
    return static_cast<Card>(rank * 4 + suit); 
}

// =============================================================================
// HAND
// =============================================================================

struct Hand {
    Card hi, lo;  // hi >= lo always
    constexpr Hand() : hi(0), lo(0) {}
    constexpr Hand(Card a, Card b) : hi(a > b ? a : b), lo(a > b ? b : a) {}
};

// =============================================================================
// DECK
// =============================================================================

struct Deck {
    static constexpr int SIZE = 52;
    static constexpr auto ORDERED = []() {
        std::array<Card, SIZE> deck{};
        for (int i = 0; i < SIZE; ++i) deck[i] = static_cast<Card>(i);
        return deck;
    }();

    std::array<Card, SIZE> cards;
    uint8_t pos[SIZE];   // pos[card] = its current index in cards[]
    int size;

    std::array<Card, SIZE> snapshot_cards;
    int snapshot_size = 0;

    Deck() noexcept { reset(); }

    void reset() noexcept {
        cards = ORDERED;
        size  = SIZE;
        for (int i = 0; i < SIZE; ++i) pos[i] = (uint8_t)i;
    }

    void remove(Card card) noexcept {
        int  i    = pos[card];
        Card last = cards[--size];
        cards[i]  = last;
        pos[last] = (uint8_t)i;
    }

    Card draw(PCG32& rng) noexcept {
        int  idx  = (int)rng.random_bounded((uint32_t)size);
        Card card = cards[idx];
        Card last = cards[--size];
        cards[idx] = last;
        pos[last]  = (uint8_t)idx;
        return card;
    }

    void save() noexcept {
        snapshot_cards = cards;
        snapshot_size  = size;
    }

    // Draw n cards from the post-remove snapshot into board[] without
    // mutating the snapshot — no restore() needed between trials.
    void draw_board(PCG32& rng, Card* board, int n) noexcept {
        Card tmp[SIZE];
        int  sz = snapshot_size;
        std::memcpy(tmp, snapshot_cards.data(), (size_t)sz);
        for (int i = 0; i < n; ++i) {
            int  idx  = (int)rng.random_bounded((uint32_t)sz);
            board[i]  = tmp[idx];
            tmp[idx]  = tmp[--sz];
        }
    }
};

// =============================================================================
// DEBUG UTILITIES
// =============================================================================

namespace Debug {

    inline void print_card(Card c) noexcept {
        static constexpr const char* RANKS[] = {
            "2","3","4","5","6","7","8","9","T","J","Q","K","A"
        };
        static constexpr const char* SUITS[] = {
            "c","d","h","s"
        };
        static constexpr const char* COLOURS[] = {
            "\033[38;2;0;200;83m",    // clubs    — green
            "\033[38;2;41;121;255m",  // diamonds — blue
            "\033[38;2;213;0;0m",     // hearts   — red
            "\033[38;2;170;0;255m"    // spades   — purple
        };
        static constexpr const char* RESET = "\033[0m";

        std::cout << COLOURS[card_suit(c)]
                  << RANKS[card_rank(c)]
                  << SUITS[card_suit(c)]
                  << RESET;
    }

    inline void print_card_ln(Card c) noexcept {
        print_card(c);
        std::cout << '\n';
    }

    inline void print_hand(const Hand& h) noexcept {
        print_card(h.hi);
        std::cout << ' ';
        print_card(h.lo);
    }

    inline void print_deck(const Deck& d, int per_row = 13) noexcept {
        for (int i = 0; i < d.size; ++i) {
            print_card(d.cards[i]);
            std::cout << ' ';
            if ((i + 1) % per_row == 0) std::cout << '\n';
        }
        if (d.size % per_row != 0) std::cout << '\n';
    }

} // namespace Debug