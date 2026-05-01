#pragma once

#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../equity_engine/Engine.hpp"

// loaded equity tables — globals, populated by load_equity()
inline std::vector<uint16_t> equity_2way;
inline std::vector<uint16_t> equity_3way;
inline std::vector<uint16_t> equity_4way;

// constexpr double EQUITY_SCALE = 65535.0;

// load all three equity tables from data/
inline void load_equity(const char* dir = "../data") {
    auto load = [](const char* path, std::vector<uint16_t>& tbl, size_t n) {
        FILE* f = std::fopen(path, "rb");
        if (!f) { std::perror(path); std::exit(1); }
        tbl.resize(n);
        std::fread(tbl.data(), sizeof(uint16_t), n, f);
        std::fclose(f);
    };

    char path[256];
    std::snprintf(path, sizeof(path), "%s/equity_2way.bin", dir);
    load(path, equity_2way, NUM_2WAY * 2);

    std::snprintf(path, sizeof(path), "%s/equity_3way.bin", dir);
    load(path, equity_3way, NUM_3WAY * 3);

    std::snprintf(path, sizeof(path), "%s/equity_4way.bin", dir);
    load(path, equity_4way, NUM_4WAY * 4);
}

// equity lookup — returns hero's equity in [0, 1]
// hero_idx is which player in the sorted combo hero is (0 to n-1)

inline double get_equity_2way(int a, int b, int hero_idx) {
    if (a > b) { std::swap(a, b); hero_idx = 1 - hero_idx; }
    uint32_t idx = combo_index(a, b);
    return equity_2way[idx * 2 + hero_idx] / EQUITY_SCALE;
}

inline double get_equity_3way(int a, int b, int c, int hero_idx) {
    // sort and track hero's new position
    int classes[3]   = {a, b, c};
    int orig_pos[3]  = {0, 1, 2};
    for (int i = 0; i < 3; ++i)
        for (int j = i + 1; j < 3; ++j)
            if (classes[i] > classes[j]) {
                std::swap(classes[i], classes[j]);
                std::swap(orig_pos[i], orig_pos[j]);
            }
    int new_hero = -1;
    for (int i = 0; i < 3; ++i) if (orig_pos[i] == hero_idx) new_hero = i;

    uint32_t idx = combo_index(classes[0], classes[1], classes[2]);
    return equity_3way[idx * 3 + new_hero] / EQUITY_SCALE;
}

inline double get_equity_4way(int a, int b, int c, int d, int hero_idx) {
    int classes[4]   = {a, b, c, d};
    int orig_pos[4]  = {0, 1, 2, 3};
    for (int i = 0; i < 4; ++i)
        for (int j = i + 1; j < 4; ++j)
            if (classes[i] > classes[j]) {
                std::swap(classes[i], classes[j]);
                std::swap(orig_pos[i], orig_pos[j]);
            }
    int new_hero = -1;
    for (int i = 0; i < 4; ++i) if (orig_pos[i] == hero_idx) new_hero = i;

    uint32_t idx = combo_index(classes[0], classes[1], classes[2], classes[3]);
    return equity_4way[idx * 4 + new_hero] / EQUITY_SCALE;
}

/*
USAGE:

load_equity();

// hero is player 0 in matchup
double eq = get_equity_2way(0, 1, 0);  // AA vs KK, hero is AA
// eq ≈ 0.82

*/