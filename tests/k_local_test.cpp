// include Google Test
#include <gtest/gtest.h>
#include <gmock/gmock.h>
// include STL
#include <iostream>
#include <utility>
#include <numeric>
#include <random>
#include <tuple>
#include <type_traits>
#include <vector>
#include <algorithm>
#include <chrono>

// include OpenJij
#include <graph/all.hpp>
#include <system/all.hpp>
#include <updater/all.hpp>
#include <algorithm/all.hpp>
#include <result/all.hpp>
#include <utility/schedule_list.hpp>
#include <utility/union_find.hpp>
#include <utility/random.hpp>
#include <utility/gpu/memory.hpp>
#include <utility/gpu/cublas.hpp>

TEST(KLocal, Constructor) {
   
   openjij::graph::Index num_spins = 3;
   openjij::graph::Polynomial<double> poly_graph(num_spins, cimod::Vartype::BINARY);
   
   poly_graph.J(   {0}   ) = +10.0;//0
   poly_graph.J(   {1}   ) = +1.0 ;//1
   poly_graph.J(   {2}   ) = +2.0 ;//2
   poly_graph.J( {0, 1}  ) = +11.0;//3
   poly_graph.J( {0, 2}  ) = +22.0;//4
   poly_graph.J( {1, 2}  ) = +12.0;//5
   poly_graph.J({0, 1, 2}) = +12.0;//6
   
   openjij::graph::Spins spin = {1, 0, 1};
   
   auto poly_system = openjij::system::make_k_local_polynomial(spin, poly_graph);
   
   
}

