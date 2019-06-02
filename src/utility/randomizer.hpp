#ifndef OPENJIJ_UTILITY_RANDOMIZER_HPP__
#define OPENJIJ_UTILITY_RANDOMIZER_HPP__

#include <random>

namespace openjij {
    namespace utility {
        template <typename RandomNumberGenerator = std::mt19937>
        struct Randomizer {
            explicit Randomizer(const size_t seed) noexcept
                : _mt{seed} {
            }

            template<typename IntType = int>
            IntType generate_uniform_int_distribution(IntType min, IntType max) {
                std::uniform_int_distribution<IntType> uid(min, max);

                return uid(_mt);
            }

            template<typename RealType = double>
            RealType generate_uniform_real_distribution(RealType min, RealType max) {
                std::uniform_real_distribution<RealType> urd(min, max);

                return urd(_mt);
            }

            private:
            RandomNumberGenerator _mt;
        };

        using RandomizerMT32 = Randomizer<std::mt19937>;
        using RandomizerMT64 = Randomizer<std::mt19937_64>;
    } // namespace utility
} // namespace openjij
#endif
