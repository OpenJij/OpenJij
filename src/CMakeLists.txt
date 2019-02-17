add_library(dense SHARED graph/dense.cpp)
add_library(sparse SHARED graph/sparse.cpp)
add_library(square SHARED graph/square.cpp)
add_library(chimera SHARED graph/chimera.cpp)

add_library(clising SHARED method/classical_ising.cpp)
add_library(qising SHARED method/quantum_ising.cpp)

add_library(sa SHARED algorithm/sa.cpp)
add_library(sqa SHARED algorithm/sqa.cpp)

target_link_libraries(clising PRIVATE sa PRIVATE sparse)
target_link_libraries(qising PRIVATE sqa PRIVATE sparse)

