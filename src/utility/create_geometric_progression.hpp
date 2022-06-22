#pragma once

namespace openjij {
    namespace utility {
        template<typename ForwardIterator, typename T>
        void make_geometric_progression(ForwardIterator first, ForwardIterator last, T value, T ratio) {
            for(;first != last; ++first) {
                *first = value;
                value *= ratio;
            }
        }
    } // namespace utility
} // namespace openjij


