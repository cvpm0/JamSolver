#include "Engine.hpp"
#include <thread>

std::vector<uint16_t> equity_2way(NUM_2WAY * 2, 0);
std::vector<uint16_t> equity_3way(NUM_3WAY * 3, 0);
std::vector<uint16_t> equity_4way(NUM_4WAY * 4, 0);

int main() {
    unsigned int n     = std::thread::hardware_concurrency() - 1;
    int          chunk = 169 / n;

    auto run = [&](auto fn, auto& table) {
        std::vector<std::thread> threads;
        for (int i = 0; i < (int)n; ++i) {
            int start = i * chunk;
            int end   = (i == (int)n - 1) ? 169 : start + chunk;
            threads.emplace_back([start, end, &table, fn]() {
                PCG32 rng;
                Deck  deck;
                fn(start, end, table, rng, deck);
            });
        }
        for (auto& t : threads) t.join();
    };

    run(run_matchups_2way, equity_2way);
    run(run_matchups_3way, equity_3way);
    run(run_matchups_4way, equity_4way);

    // 2-way validation
    uint32_t aa_kk   = combo_index(0, 1);
    uint32_t aa_aa   = combo_index(0, 0);
    uint32_t aks_72o = combo_index(13, 168);
    uint32_t two_ako = combo_index(12, 91);

    std::cout << "=== 2-way ===\n";
    std::cout << "AA vs KK:   " << equity_2way[aa_kk   * 2 + 0] / EQUITY_SCALE << "\n";
    std::cout << "KK vs AA:   " << equity_2way[aa_kk   * 2 + 1] / EQUITY_SCALE << "\n";
    std::cout << "AA vs AA:   " << equity_2way[aa_aa   * 2 + 0] / EQUITY_SCALE << "\n";
    std::cout << "AKs vs 72o: " << equity_2way[aks_72o * 2 + 0] / EQUITY_SCALE << "\n";
    std::cout << "22 vs AKo:  " << equity_2way[two_ako * 2 + 0] / EQUITY_SCALE << "\n";

    // 3-way validation — AA ~63%, KK ~20%, QQ ~17%
    uint32_t aa_kk_qq = combo_index(0, 1, 2);
    uint32_t aa_kk_kk = combo_index(0, 1, 1);

    std::cout << "\n=== 3-way ===\n";
    std::cout << "AA vs KK vs QQ:\n";
    std::cout << "  AA: " << equity_3way[aa_kk_qq * 3 + 0] / EQUITY_SCALE << "\n";
    std::cout << "  KK: " << equity_3way[aa_kk_qq * 3 + 1] / EQUITY_SCALE << "\n";
    std::cout << "  QQ: " << equity_3way[aa_kk_qq * 3 + 2] / EQUITY_SCALE << "\n";
    std::cout << "AA vs KK vs KK:\n";
    std::cout << "  AA: " << equity_3way[aa_kk_kk * 3 + 0] / EQUITY_SCALE << "\n";
    std::cout << "  KK: " << equity_3way[aa_kk_kk * 3 + 1] / EQUITY_SCALE << "\n";
    std::cout << "  KK: " << equity_3way[aa_kk_kk * 3 + 2] / EQUITY_SCALE << "\n";

    // 4-way validation — AA ~63%, KK ~20%, QQ ~11%, JJ ~6%
    uint32_t aa_kk_qq_jj = combo_index(0, 1, 2, 3);

    std::cout << "\n=== 4-way ===\n";
    std::cout << "AA vs KK vs QQ vs JJ:\n";
    std::cout << "  AA: " << equity_4way[aa_kk_qq_jj * 4 + 0] / EQUITY_SCALE << "\n";
    std::cout << "  KK: " << equity_4way[aa_kk_qq_jj * 4 + 1] / EQUITY_SCALE << "\n";
    std::cout << "  QQ: " << equity_4way[aa_kk_qq_jj * 4 + 2] / EQUITY_SCALE << "\n";
    std::cout << "  JJ: " << equity_4way[aa_kk_qq_jj * 4 + 3] / EQUITY_SCALE << "\n";
}
