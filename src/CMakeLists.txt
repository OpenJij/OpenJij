add_library(updater SHARED updater/single_spin_flip.cpp)

add_library(sampler SHARED sampler/sampler.cpp)
target_link_libraries(sampler PRIVATE updater)

