//
//  test_ksuzuki.cpp
//  OpenJijXcode
//
//  Created by 鈴木浩平 on 2021/03/01.
//


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

#include "./testcase.hpp"

static openjij::utility::ClassicalScheduleList generate_schedule_list() {
    return openjij::utility::make_classical_schedule_list(1, 1000.0, 100, 100);
}

TEST(Poly_Graph, ConstructorCimod) {
   
   cimod::Polynomial<openjij::graph::Index, double> Polynomial {
      {{0}, 0.0}, {{1}, 1.0}, {{2}, 2.0},
      {{0, 1}, 11.0}, {{0, 2}, 22.0}, {{1, 2}, 12.0},
      {{0, 1, 2}, +12}
   };
   cimod::Vartype vartype = cimod::Vartype::SPIN;
   cimod::BinaryPolynomialModel<openjij::graph::Index, double> bpm_cimod(Polynomial, vartype);
   
   openjij::graph::Polynomial<double> poly_graph(bpm_cimod);

   EXPECT_EQ(poly_graph.GetMaxVariable(), 2);

   EXPECT_EQ(bpm_cimod.get_polynomial().size(), poly_graph.GetInteractions().size());

   for (const auto &it: Polynomial) {
      EXPECT_DOUBLE_EQ(bpm_cimod.get_polynomial().at(it.first), poly_graph.GetInteractions().at(it.first));
   }
   
   EXPECT_DOUBLE_EQ(bpm_cimod.get_polynomial().at(   {0}   ), poly_graph.J(   {0}   ));
   EXPECT_DOUBLE_EQ(bpm_cimod.get_polynomial().at(   {1}   ), poly_graph.J(   {1}   ));
   EXPECT_DOUBLE_EQ(bpm_cimod.get_polynomial().at(   {2}   ), poly_graph.J(   {2}   ));
   EXPECT_DOUBLE_EQ(bpm_cimod.get_polynomial().at( {0, 1}  ), poly_graph.J( {0, 1}  ));
   EXPECT_DOUBLE_EQ(bpm_cimod.get_polynomial().at( {0, 2}  ), poly_graph.J( {0, 2}  ));
   EXPECT_DOUBLE_EQ(bpm_cimod.get_polynomial().at( {1, 2}  ), poly_graph.J( {1, 2}  ));
   EXPECT_DOUBLE_EQ(bpm_cimod.get_polynomial().at({0, 1, 2}), poly_graph.J({0, 1, 2}));
   
   EXPECT_DOUBLE_EQ(bpm_cimod.get_polynomial().at(   {0}   ), poly_graph.J(0));
   EXPECT_DOUBLE_EQ(bpm_cimod.get_polynomial().at(   {1}   ), poly_graph.J(1));
   EXPECT_DOUBLE_EQ(bpm_cimod.get_polynomial().at(   {2}   ), poly_graph.J(2));
   EXPECT_DOUBLE_EQ(bpm_cimod.get_polynomial().at( {0, 1}  ), poly_graph.J(0,1));
   EXPECT_DOUBLE_EQ(bpm_cimod.get_polynomial().at( {0, 2}  ), poly_graph.J(0,2));
   EXPECT_DOUBLE_EQ(bpm_cimod.get_polynomial().at( {1, 2}  ), poly_graph.J(1,2));
   EXPECT_DOUBLE_EQ(bpm_cimod.get_polynomial().at({0, 1, 2}), poly_graph.J(0,1,2));
   
}

TEST(Poly_Graph, ConstructorJson) {
   
   cimod::Polynomial<std::string, double> Polynomial {
      {{"a"}, 0.0}, {{"b"}, 1.0}, {{"c"}, 2.0},
      {{"a", "b"}, 11.0}, {{"a", "c"}, 22.0}, {{"b", "c"}, 12.0},
      {{"a", "b", "c"}, +12}
   };
   cimod::Vartype vartype = cimod::Vartype::SPIN;
   cimod::BinaryPolynomialModel<std::string, double> bpm_cimod(Polynomial, vartype);
   
   openjij::graph::Polynomial<double> poly_graph(bpm_cimod.to_serializable());

   EXPECT_EQ(poly_graph.GetMaxVariable(), 2);

   EXPECT_EQ(bpm_cimod.get_polynomial().size(), poly_graph.GetInteractions().size());

   EXPECT_DOUBLE_EQ(bpm_cimod.get_polynomial().at(   {"a"}   )    , poly_graph.J(   {0}   ));
   EXPECT_DOUBLE_EQ(bpm_cimod.get_polynomial().at(   {"b"}   )    , poly_graph.J(   {1}   ));
   EXPECT_DOUBLE_EQ(bpm_cimod.get_polynomial().at(   {"c"}   )    , poly_graph.J(   {2}   ));
   EXPECT_DOUBLE_EQ(bpm_cimod.get_polynomial().at( {"a", "b"}  )  , poly_graph.J( {0, 1}  ));
   EXPECT_DOUBLE_EQ(bpm_cimod.get_polynomial().at( {"a", "c"}  )  , poly_graph.J( {0, 2}  ));
   EXPECT_DOUBLE_EQ(bpm_cimod.get_polynomial().at( {"b", "c"}  )  , poly_graph.J( {1, 2}  ));
   EXPECT_DOUBLE_EQ(bpm_cimod.get_polynomial().at({"a", "b", "c"}), poly_graph.J({0, 1, 2}));
   
}

TEST(Poly_Graph, AddInteractions) {
   
   openjij::graph::Index num_spins = 3;
   openjij::graph::Polynomial<double> poly_graph(num_spins);
   
   poly_graph.J(   {0}   ) = +0.0 ;
   poly_graph.J(   {1}   ) = +1.0 ;
   poly_graph.J(   {2}   ) = +2.0 ;
   poly_graph.J( {0, 1}  ) = +11.0;
   poly_graph.J( {0, 2}  ) = +22.0;
   poly_graph.J( {1, 2}  ) = +12.0;
   poly_graph.J({0, 1, 2}) = +12.0;
   
   EXPECT_EQ(poly_graph.GetMaxVariable(), 2);
   EXPECT_EQ(poly_graph.GetInteractions().size(), 7);
   
   EXPECT_DOUBLE_EQ(poly_graph.J(   {0}   ), +0.0 );
   EXPECT_DOUBLE_EQ(poly_graph.J(   {1}   ), +1.0 );
   EXPECT_DOUBLE_EQ(poly_graph.J(   {2}   ), +2.0 );
   EXPECT_DOUBLE_EQ(poly_graph.J( {0, 1}  ), +11.0);
   EXPECT_DOUBLE_EQ(poly_graph.J( {0, 2}  ), +22.0);
   EXPECT_DOUBLE_EQ(poly_graph.J( {1, 2}  ), +12.0);
   EXPECT_DOUBLE_EQ(poly_graph.J({0, 1, 2}), +12.0);
   
   poly_graph.J(0)     += +0.0 ;
   poly_graph.J(1)     += +1.0 ;
   poly_graph.J(2)     += +2.0 ;
   poly_graph.J(0,1)   += +11.0;
   poly_graph.J(0,2)   += +22.0;
   poly_graph.J(1,2)   += +12.0;
   poly_graph.J(0,1,2) += +12.0;
   
   EXPECT_DOUBLE_EQ(poly_graph.J(   {0}   ), +0.0 *2);
   EXPECT_DOUBLE_EQ(poly_graph.J(   {1}   ), +1.0 *2);
   EXPECT_DOUBLE_EQ(poly_graph.J(   {2}   ), +2.0 *2);
   EXPECT_DOUBLE_EQ(poly_graph.J( {0, 1}  ), +11.0*2);
   EXPECT_DOUBLE_EQ(poly_graph.J( {0, 2}  ), +22.0*2);
   EXPECT_DOUBLE_EQ(poly_graph.J( {1, 2}  ), +12.0*2);
   EXPECT_DOUBLE_EQ(poly_graph.J({0, 1, 2}), +12.0*2);

}

TEST(Poly_Graph, Energy) {
   
   cimod::Polynomial<openjij::graph::Index, double> Polynomial {
      {{0}, 0.0}, {{1}, 1.0}, {{2}, 2.0},
      {{0, 1}, 11.0}, {{0, 2}, 22.0}, {{1, 2}, 12.0},
      {{0, 1, 2}, +12}
   };
   cimod::Vartype vartype = cimod::Vartype::SPIN;
   cimod::BinaryPolynomialModel<openjij::graph::Index, double> bpm_cimod(Polynomial, vartype);
   
   openjij::graph::Polynomial<double> poly_graph(bpm_cimod);
   
   openjij::graph::Spins spin = {+1, -1, +1};
   cimod::Sample<openjij::graph::Index> spin_for_cimod {
      {0, +1}, {1, -1}, {2, +1}
   };

   EXPECT_DOUBLE_EQ(bpm_cimod.energy(spin_for_cimod), poly_graph.CalclateEnergy(spin));
   
}
