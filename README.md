# JamSolver

[**Try it →**](https://cvpm0.github.io/JamSolver/)

Note: Best on mobile or portrait tab

A preflop jam/fold solver for 4-player short-stack poker. Given a position and action history, it outputs a virtually unexploitable strategy — the jam frequency that no opponent adjustment can meaningfully profit against.

The project splits into two executables sharing a data layer: an **equity engine** that precomputes all-pairs matchup tables via Monte Carlo, and a **CFR+ solver** that finds equilibrium over a 14-state game tree using those tables as O(1) lookups. The solver output is consumed by a lightweight web frontend for table-side reference.

## Equity Engine

The engine precomputes equity tables for every matchup of 169 hand classes — 14,365 two-way, 815,815 three-way — and writes them to flat binary files. The solver loads these at startup and indexes into them directly; no equity computation happens during solving.

### Hand Evaluation

`evaluate7` encodes hand strength as a packed `uint32_t` where higher integer = stronger hand, so comparison across all hand categories is a single `>` with no branching:

```
(category << 20) | (r1 << 16) | (r2 << 12) | (r3 << 8) | (r4 << 4) | r5
```

The evaluator classifies a 7-card hand in a single pass using bitmask state transitions rather than a rank-count array. As each card is processed, its rank bit propagates through four masks — `rank_mask`, `rank_2plus`, `rank_3plus`, `rank_4` — via conditional OR. This avoids a 13-element frequency array and the second pass to interpret it:

```cpp
if (rank_mask & bit) {
    if (rank_2plus & bit) {
        if (rank_3plus & bit) rank_4     |= bit;
        else                  rank_3plus |= bit;
    } else                    rank_2plus |= bit;
}
rank_mask |= bit;
```

The four masks encode the full multiplicity structure in 64 bits of register state. Category detection is then a sequence of popcount/clz tests on the masks — no iteration over ranks.

Straight detection uses the shift-AND pattern. Five consecutive rank bits survive this filter only if all five are set:

```cpp
uint16_t c = mask & (mask >> 1) & (mask >> 2) & (mask >> 3) & (mask >> 4);
```

The highest surviving bit (via `__builtin_clz`) gives the straight's top rank. The wheel (A-2-3-4-5) is a special-case bitmask check. All kicker extraction throughout the evaluator uses `__builtin_clz` on the residual mask after removing used ranks — no linear scan over 13 ranks.

### Canonicalisation and Collision Avoidance

A hand class like AKs doesn't specify suits — it could be any of four suit assignments. When setting up a multi-way simulation, concrete suits must be assigned to each player without card collisions. The engine resolves this with a `uint64_t` bitmask over the 52-card space: each candidate suit assignment is tested with a single AND against the occupied mask, and conflicts advance an odometer over a compile-time combo table (`constexpr` lambda, ~4.4 KB, L1-resident).

### Deck and Sampling

The deck maintains a parallel position array (`pos[card] = index`) alongside the card array, enabling O(1) removal via swap-to-back without a linear search:

```cpp
void remove(Card card) noexcept {
    int  i    = pos[card];
    Card last = cards[--size];
    cards[i]  = last;
    pos[last] = (uint8_t)i;
}
```

Board sampling across Monte Carlo trials uses a snapshot mechanism: after removing all hole cards, the deck state is saved once. Each trial draws from a stack-allocated copy of the snapshot rather than restoring the deck, eliminating a memcpy-per-trial of the full deck state.

### Parallelism

The 169 hero classes are partitioned across `std::thread` workers with per-thread `Deck` instances and a from-scratch `PCG32` implementation. No shared mutable state exists in the hot path — no atomics, no locks, no false sharing. The RNG implements Lemire's nearly-divisionless method for unbiased bounded sampling, with each thread's stream seeded from hardware entropy with a forced-odd increment to guarantee full period.

## Solver

### Game Tree

The game models 4 positions with binary actions (jam/fold), producing 14 reachable information states. EV computation walks forward from a decision node, weighting each branch by the product of downstream opponents' jam probabilities from their current strategies.

### Equity Cache

A naïve EV computation would sum over all 169 villain hands per showdown node — O(169) for heads-up, O(169²) for three-way — and this runs inside a loop over 14 states × 169 hero hands per iteration. The dominant cost.

Instead, after each strategy update the solver fills two caches indexed by hero hand:

- `eq_vs_one[state][hero]` — equity against one villain's weighted jam range
- `eq_vs_two[state1][state2][hero]` — equity against two independent jam ranges

EV functions then read cached values at O(1). Only the 11 `(state1, state2)` pairs actually referenced by the game tree are filled — determined by static analysis of the EV functions.

After convergence, jam ranges are sparse (typically 5–30 of 169 classes have nonzero weight). The cache fill exploits this by compacting nonzero-weight indices into a dense array before the inner loop:

```cpp
int idx[NUM_HANDS], n = 0;
for (int v = 0; v < NUM_HANDS; ++v)
    if (w[v] > 0.0) idx[n++] = v;

// Inner loop: n iterations, not 169
for (int i = 0; i < n; ++i) { ... }
```

For the three-way cache this reduces the inner work from 169² ≈ 28k iterations to roughly 25² = 625 in late iterations — a 45× reduction that compounds across the 11 pairs × 169 hero hands evaluated each iteration. The compacted loop also eliminates branch misprediction overhead from `continue`-based zero-skipping, since every iteration does real work.

Both cache fills are parallelised with OpenMP (`schedule(dynamic)` to handle variable cost across pairs with different range densities). Each parallel unit writes to a disjoint cache slice, so no synchronisation is needed.

### CFR+

The solver uses CFR+ with linear strategy averaging. Regrets are floored at zero on every update (rather than at strategy-read time as in vanilla CFR), and each iteration contributes to the strategy average with weight proportional to the iteration index. The triangular normalization sum uses `int64_t` arithmetic — at 100,000 iterations the product `N × (N+1)` exceeds `INT32_MAX`, an overflow that silently corrupts the output if computed in 32-bit integers.

Four-way pots (when all three opponents jam) are approximated using three-way equity scaled by a constant factor, since true four-way equity lookup would require a 169⁴ table and the all-jam state is reached rarely enough in equilibrium that the approximation has negligible impact on upstream strategies.

### Output

Converged strategies are serialised to a `.jams` binary: a 16-byte header followed by `double[14][169]` row-major. Total file: 18,944 bytes. The web frontend reads a single `(state, hand)` entry at a known offset — no parsing, no deserialisation.
