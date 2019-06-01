#ifndef OPENJIJ_SYSTEM_SYSTEM_TRAITS_HPP__
#define OPENJIJ_SYSTEM_SYSTEM_TRAITS_HPP__

namespace openjij {
    namespace system {
        struct classical_system {};
        struct quantum_system {};

        template<typename System>
        struct get_system_type {
            using type = typename System::system_type;
        };
    } // namespace system
} // namespace openjij

#endif
