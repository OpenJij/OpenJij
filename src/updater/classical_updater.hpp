#ifndef OPENJIJ_UPDATER_CLASSICALUPDATER_HPP__
#define OPENJIJ_UPDATER_CLASSICALUPDATER_HPP__

namespace openjij {
    namespace updater {
        template<typename Derived>
        struct ClassicalUpdater {
            double update(const double beta, const std::string& algo = "") {
                static_cast<const Derived&>(*this).update();
            }
        };
    } // namespace updater
} // namespace openjij

#endif
