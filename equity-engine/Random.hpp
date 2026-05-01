#pragma once


#include <cstdint>
#include <random>

// reference: https://github.com/wjakob/pcg32 (unofficial) 

struct PCG32 {
    private:
        static constexpr uint64_t MULTIPLIER = 6364136223846793005ULL;

        uint64_t state;          // RNG internal state
        uint64_t increment;  // stream selector (must be odd)

        uint32_t next() noexcept {
            uint64_t old = state;
            state = old * MULTIPLIER + increment;

            uint32_t xorshifted = ((old >> 18u) ^ old) >> 27u;
            uint32_t rot = old >> 59u;
            return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
        }
        
    
    public:
        PCG32() noexcept {
            std::random_device rd;
            increment = ((uint64_t)rd() << 33) | 1u;  // odd, full entropy
            state     = (uint64_t)rd() << 32 | rd();  // full 64-bit entropy
            next();                                    // warm up
        }

        // TODO: add explicit PCG32(uint64_t seed) constructor for reproducible runs (debugging/benchmarking)

        

        uint32_t random_bounded(uint32_t bound) noexcept { // can we guarantee uniform distribution on bound
            uint32_t threshold = -bound % bound;

            while (true) {
                uint32_t r = next();
                if (r >= threshold) {
                    return r % bound;
                }
            }
        }      

};