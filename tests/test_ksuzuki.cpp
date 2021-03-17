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

static constexpr std::size_t num_system_size = 8;
#define TEST_CASE_INDEX 1
#include "./testcase.hpp"

static openjij::utility::ClassicalScheduleList generate_schedule_list(){
    return openjij::utility::make_classical_schedule_list(0.1, 100.0, 100, 100);
}

TEST(PolyGraph, ConstructorCimod1) {
   
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

TEST(PolyGraph, ConstructorCimod2) {
   
   cimod::Polynomial<openjij::graph::Index, double> Polynomial {
      {{0}, 0.0}, {{1}, 1.0}, {{2}, 2.0},
      {{0, 1}, 11.0}, {{1, 0}, 11.0}, {{0, 2}, 22.0}, {{2, 0}, 22.0}, {{1, 2}, 12.0}, {{2, 1}, 12.0},
      {{0, 1, 2}, +12}, {{0, 2, 1}, +12}, {{1, 0, 2}, +12}, {{1, 2, 0}, +12},
      {{2, 0, 1}, +12}, {{2, 1, 0}, +12}
   };
   
   cimod::Vartype vartype = cimod::Vartype::SPIN;
   cimod::BinaryPolynomialModel<openjij::graph::Index, double> bpm_cimod(Polynomial, vartype);
   
   openjij::graph::Polynomial<double> poly_graph(bpm_cimod);

   EXPECT_EQ(poly_graph.GetMaxVariable(), 2);

   EXPECT_EQ(bpm_cimod.get_polynomial().size(), poly_graph.GetInteractions().size() + 8);
   
   EXPECT_DOUBLE_EQ(poly_graph.J(   {0}   ), bpm_cimod.get_polynomial().at(   {0}   ));
   EXPECT_DOUBLE_EQ(poly_graph.J(   {1}   ), bpm_cimod.get_polynomial().at(   {1}   ));
   EXPECT_DOUBLE_EQ(poly_graph.J(   {2}   ), bpm_cimod.get_polynomial().at(   {2}   ));
   EXPECT_DOUBLE_EQ(poly_graph.J( {0, 1}  ), bpm_cimod.get_polynomial().at( {0, 1}  )*2);
   EXPECT_DOUBLE_EQ(poly_graph.J( {0, 2}  ), bpm_cimod.get_polynomial().at( {0, 2}  )*2);
   EXPECT_DOUBLE_EQ(poly_graph.J( {1, 2}  ), bpm_cimod.get_polynomial().at( {1, 2}  )*2);
   EXPECT_DOUBLE_EQ(poly_graph.J({0, 1, 2}), bpm_cimod.get_polynomial().at({0, 1, 2})*6);
   
}

TEST(PolyGraph, ConstructorJson) {
   
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

TEST(PolyGraph, AddInteractions) {
   
   openjij::graph::Index num_spins = 3;
   openjij::graph::Polynomial<double> poly_graph(num_spins, "SPIN");
   
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
   
   poly_graph.J(0,0,0) += +0.0 ;
   poly_graph.J(1,1,1) += +1.0 ;
   poly_graph.J(2,2,2) += +2.0 ;
   poly_graph.J(0,1,1) += +11.0;
   poly_graph.J(0,2,2) += +22.0;
   poly_graph.J(1,2,1) += +12.0;
   poly_graph.J(0,1,2) += +12.0;
   
   EXPECT_DOUBLE_EQ(poly_graph.J(   {0}   ), +0.0 *3);
   EXPECT_DOUBLE_EQ(poly_graph.J(   {1}   ), +1.0 *3);
   EXPECT_DOUBLE_EQ(poly_graph.J(   {2}   ), +2.0 *3);
   EXPECT_DOUBLE_EQ(poly_graph.J( {0, 1}  ), +11.0*3);
   EXPECT_DOUBLE_EQ(poly_graph.J( {0, 2}  ), +22.0*3);
   EXPECT_DOUBLE_EQ(poly_graph.J( {1, 2}  ), +12.0*3);
   EXPECT_DOUBLE_EQ(poly_graph.J({0, 1, 2}), +12.0*3);

}

TEST(PolyGraph, Energy) {
   
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

   EXPECT_DOUBLE_EQ(bpm_cimod.energy(spin_for_cimod), poly_graph.CalculateEnergy(spin));
   
}

TEST(PolySystem, ConstructorSpin1) {
   
   openjij::graph::Index num_spins = 3;
   openjij::graph::Polynomial<double> poly_graph(num_spins, "SPIN");
   
   poly_graph.J(   {0}   ) = +0.0 ;//0
   poly_graph.J(   {1}   ) = +1.0 ;//1
   poly_graph.J(   {2}   ) = +2.0 ;//2
   poly_graph.J( {0, 1}  ) = +11.0;//3
   poly_graph.J( {0, 2}  ) = +22.0;//4
   poly_graph.J( {1, 2}  ) = +12.0;//5
   poly_graph.J({0, 1, 2}) = +12.0;//6
   
   openjij::graph::Spins spin = {+1, -1, +1};
   
   openjij::system::ClassicalIsingPolynomial<openjij::graph::Polynomial<double>> poly_system = openjij::system::make_classical_ising_polynomial(spin, poly_graph);
   
   EXPECT_EQ(poly_system.num_spins, 3);
   for (auto i = 0; i < poly_system.num_spins; ++i) {
      EXPECT_EQ(poly_system.spin[i], spin[i]);
   }
   
   //Check J_term: set in SetJTerm()
   EXPECT_DOUBLE_EQ(poly_system.J_term[0], +0 );
   EXPECT_DOUBLE_EQ(poly_system.J_term[1], -1 );
   EXPECT_DOUBLE_EQ(poly_system.J_term[2], +2 );
   EXPECT_DOUBLE_EQ(poly_system.J_term[3], -11);
   EXPECT_DOUBLE_EQ(poly_system.J_term[4], +22);
   EXPECT_DOUBLE_EQ(poly_system.J_term[5], -12);
   EXPECT_DOUBLE_EQ(poly_system.J_term[6], -12);
   
   //Check connected_J_term_index: set in SetJTerm()
   EXPECT_EQ(poly_system.connected_J_term_index.size(), poly_system.num_spins);
   EXPECT_EQ(poly_system.connected_J_term_index[0].size(), 4);
   EXPECT_EQ(poly_system.connected_J_term_index[1].size(), 4);
   EXPECT_EQ(poly_system.connected_J_term_index[2].size(), 4);
   
   EXPECT_EQ(poly_system.connected_J_term_index[0][0], 0);
   EXPECT_EQ(poly_system.connected_J_term_index[0][1], 3);
   EXPECT_EQ(poly_system.connected_J_term_index[0][2], 4);
   EXPECT_EQ(poly_system.connected_J_term_index[0][3], 6);

   EXPECT_EQ(poly_system.connected_J_term_index[1][0], 1);
   EXPECT_EQ(poly_system.connected_J_term_index[1][1], 3);
   EXPECT_EQ(poly_system.connected_J_term_index[1][2], 5);
   EXPECT_EQ(poly_system.connected_J_term_index[1][3], 6);
   
   EXPECT_EQ(poly_system.connected_J_term_index[2][0], 2);
   EXPECT_EQ(poly_system.connected_J_term_index[2][1], 4);
   EXPECT_EQ(poly_system.connected_J_term_index[2][2], 5);
   EXPECT_EQ(poly_system.connected_J_term_index[2][3], 6);
   
   //Check dE: set in SetdE()
   EXPECT_DOUBLE_EQ(poly_system.dE.size(), poly_system.num_spins);
   EXPECT_DOUBLE_EQ(poly_system.dE[0], -2*spin[0]*(poly_graph.J(0) + poly_graph.J(0, 1)*spin[1] + poly_graph.J(0, 2)*spin[2] + poly_graph.J(0, 1, 2)*spin[1]*spin[2]));
   EXPECT_DOUBLE_EQ(poly_system.dE[1], -2*spin[1]*(poly_graph.J(1) + poly_graph.J(0, 1)*spin[0] + poly_graph.J(1, 2)*spin[2] + poly_graph.J(0, 1, 2)*spin[0]*spin[2]));
   EXPECT_DOUBLE_EQ(poly_system.dE[2], -2*spin[2]*(poly_graph.J(2) + poly_graph.J(0, 2)*spin[0] + poly_graph.J(1, 2)*spin[1] + poly_graph.J(0, 1, 2)*spin[0]*spin[1]));

   //Check UpdateMatrix: set in SetUpdateMatrix()
   EXPECT_EQ(poly_system.row.size(), poly_system.num_spins + 1);
   EXPECT_EQ(poly_system.col.size(), 12);
   EXPECT_EQ(poly_system.val_p_spin.size(), 12);
   
   EXPECT_EQ(poly_system.row[0 ], 0 );
   EXPECT_EQ(poly_system.row[1 ], 4 );
   EXPECT_EQ(poly_system.row[2 ], 8 );
   EXPECT_EQ(poly_system.row[3 ], 12);
   
   EXPECT_EQ(poly_system.col[0 ], 1);
   EXPECT_EQ(poly_system.col[1 ], 1);
   EXPECT_EQ(poly_system.col[2 ], 2);
   EXPECT_EQ(poly_system.col[3 ], 2);
   EXPECT_EQ(poly_system.col[4 ], 0);
   EXPECT_EQ(poly_system.col[5 ], 0);
   EXPECT_EQ(poly_system.col[6 ], 2);
   EXPECT_EQ(poly_system.col[7 ], 2);
   EXPECT_EQ(poly_system.col[8 ], 0);
   EXPECT_EQ(poly_system.col[9 ], 0);
   EXPECT_EQ(poly_system.col[10], 1);
   EXPECT_EQ(poly_system.col[11], 1);
   
   EXPECT_DOUBLE_EQ(*poly_system.val_p_spin[0 ], poly_graph.J(0, 1)   *spin[0]*spin[1]        );
   EXPECT_DOUBLE_EQ(*poly_system.val_p_spin[1 ], poly_graph.J(0, 1, 2)*spin[0]*spin[1]*spin[2]);
   EXPECT_DOUBLE_EQ(*poly_system.val_p_spin[2 ], poly_graph.J(0, 2)   *spin[0]*spin[2]        );
   EXPECT_DOUBLE_EQ(*poly_system.val_p_spin[3 ], poly_graph.J(0, 1, 2)*spin[0]*spin[1]*spin[2]);
   EXPECT_DOUBLE_EQ(*poly_system.val_p_spin[4 ], poly_graph.J(0, 1)   *spin[0]*spin[1]        );
   EXPECT_DOUBLE_EQ(*poly_system.val_p_spin[5 ], poly_graph.J(0, 1, 2)*spin[0]*spin[1]*spin[2]);
   EXPECT_DOUBLE_EQ(*poly_system.val_p_spin[6 ], poly_graph.J(1, 2)   *spin[1]*spin[2]        );
   EXPECT_DOUBLE_EQ(*poly_system.val_p_spin[7 ], poly_graph.J(0, 1, 2)*spin[0]*spin[1]*spin[2]);
   EXPECT_DOUBLE_EQ(*poly_system.val_p_spin[8 ], poly_graph.J(0, 2)   *spin[0]*spin[2]        );
   EXPECT_DOUBLE_EQ(*poly_system.val_p_spin[9 ], poly_graph.J(0, 1, 2)*spin[0]*spin[1]*spin[2]);
   EXPECT_DOUBLE_EQ(*poly_system.val_p_spin[10], poly_graph.J(1, 2)   *spin[1]*spin[2]        );
   EXPECT_DOUBLE_EQ(*poly_system.val_p_spin[11], poly_graph.J(0, 1, 2)*spin[0]*spin[1]*spin[2]);
   
   //Check vartype
   EXPECT_TRUE(poly_system.isIsing);
   
   //Check variables for binary
   EXPECT_EQ(poly_system.val_binary.size(), 0);
   EXPECT_EQ(poly_system.zero_count_p_binary.size(), 0);
   EXPECT_EQ(poly_system.zero_count_binary.size(), 0);
   
}

TEST(PolySystem, ConstructorSpin2) {
   
   openjij::graph::Index num_spins = 3;
   openjij::graph::Polynomial<double> poly_graph(num_spins, "SPIN");
   
   //The spin index does not start with 0
   poly_graph.J(    {10}    ) = +0.0 ;//0
   poly_graph.J(    {11}    ) = +1.0 ;//1
   poly_graph.J(    {12}    ) = +2.0 ;//2
   poly_graph.J(  {10, 11}  ) = +11.0;//3
   poly_graph.J(  {10, 12}  ) = +22.0;//4
   poly_graph.J(  {11, 12}  ) = +12.0;//5
   poly_graph.J({10, 11, 12}) = +12.0;//6
   
   openjij::graph::Spins spin = {+1, -1, +1};
   
   openjij::system::ClassicalIsingPolynomial<openjij::graph::Polynomial<double>> poly_system = openjij::system::make_classical_ising_polynomial(spin, poly_graph);
   
   EXPECT_EQ(poly_system.num_spins, 3);
   for (auto i = 0; i < poly_system.num_spins; ++i) {
      EXPECT_EQ(poly_system.spin[i], spin[i]);
   }
   
   //Check J_term: set in SetJTerm()
   EXPECT_EQ(poly_system.J_term.size(), 7);
   EXPECT_DOUBLE_EQ(poly_system.J_term[0], +0 );
   EXPECT_DOUBLE_EQ(poly_system.J_term[1], -1 );
   EXPECT_DOUBLE_EQ(poly_system.J_term[2], +2 );
   EXPECT_DOUBLE_EQ(poly_system.J_term[3], -11);
   EXPECT_DOUBLE_EQ(poly_system.J_term[4], +22);
   EXPECT_DOUBLE_EQ(poly_system.J_term[5], -12);
   EXPECT_DOUBLE_EQ(poly_system.J_term[6], -12);
   
   //Check connected_J_term_index: set in SetJTerm()
   EXPECT_EQ(poly_system.connected_J_term_index.size(), poly_system.num_spins);
   EXPECT_EQ(poly_system.connected_J_term_index[0].size(), 4);
   EXPECT_EQ(poly_system.connected_J_term_index[1].size(), 4);
   EXPECT_EQ(poly_system.connected_J_term_index[2].size(), 4);
   
   EXPECT_EQ(poly_system.connected_J_term_index[0][0], 0);
   EXPECT_EQ(poly_system.connected_J_term_index[0][1], 3);
   EXPECT_EQ(poly_system.connected_J_term_index[0][2], 4);
   EXPECT_EQ(poly_system.connected_J_term_index[0][3], 6);

   EXPECT_EQ(poly_system.connected_J_term_index[1][0], 1);
   EXPECT_EQ(poly_system.connected_J_term_index[1][1], 3);
   EXPECT_EQ(poly_system.connected_J_term_index[1][2], 5);
   EXPECT_EQ(poly_system.connected_J_term_index[1][3], 6);
   
   EXPECT_EQ(poly_system.connected_J_term_index[2][0], 2);
   EXPECT_EQ(poly_system.connected_J_term_index[2][1], 4);
   EXPECT_EQ(poly_system.connected_J_term_index[2][2], 5);
   EXPECT_EQ(poly_system.connected_J_term_index[2][3], 6);
   
   //Check dE: set in SetdE()
   EXPECT_DOUBLE_EQ(poly_system.dE.size(), poly_system.num_spins);
   EXPECT_DOUBLE_EQ(poly_system.dE[0], -2*spin[0]*(poly_graph.J(10) + poly_graph.J(10, 11)*spin[1] + poly_graph.J(10, 12)*spin[2] + poly_graph.J(10, 11, 12)*spin[1]*spin[2]));
   EXPECT_DOUBLE_EQ(poly_system.dE[1], -2*spin[1]*(poly_graph.J(11) + poly_graph.J(10, 11)*spin[0] + poly_graph.J(11, 12)*spin[2] + poly_graph.J(10, 11, 12)*spin[0]*spin[2]));
   EXPECT_DOUBLE_EQ(poly_system.dE[2], -2*spin[2]*(poly_graph.J(12) + poly_graph.J(10, 12)*spin[0] + poly_graph.J(11, 12)*spin[1] + poly_graph.J(10, 11, 12)*spin[0]*spin[1]));

   //Check UpdateMatrix: set in SetUpdateMatrix()
   EXPECT_EQ(poly_system.row.size(), poly_system.num_spins + 1);
   EXPECT_EQ(poly_system.col.size(), 12);
   EXPECT_EQ(poly_system.val_p_spin.size(), 12);
   
   EXPECT_EQ(poly_system.row[0 ], 0 );
   EXPECT_EQ(poly_system.row[1 ], 4 );
   EXPECT_EQ(poly_system.row[2 ], 8 );
   EXPECT_EQ(poly_system.row[3 ], 12);
   
   EXPECT_EQ(poly_system.col[0 ], 1);
   EXPECT_EQ(poly_system.col[1 ], 1);
   EXPECT_EQ(poly_system.col[2 ], 2);
   EXPECT_EQ(poly_system.col[3 ], 2);
   EXPECT_EQ(poly_system.col[4 ], 0);
   EXPECT_EQ(poly_system.col[5 ], 0);
   EXPECT_EQ(poly_system.col[6 ], 2);
   EXPECT_EQ(poly_system.col[7 ], 2);
   EXPECT_EQ(poly_system.col[8 ], 0);
   EXPECT_EQ(poly_system.col[9 ], 0);
   EXPECT_EQ(poly_system.col[10], 1);
   EXPECT_EQ(poly_system.col[11], 1);
   
   EXPECT_DOUBLE_EQ(*poly_system.val_p_spin[0 ], poly_graph.J(10, 11)    *spin[0]*spin[1]        );
   EXPECT_DOUBLE_EQ(*poly_system.val_p_spin[1 ], poly_graph.J(10, 11, 12)*spin[0]*spin[1]*spin[2]);
   EXPECT_DOUBLE_EQ(*poly_system.val_p_spin[2 ], poly_graph.J(10, 12)    *spin[0]*spin[2]        );
   EXPECT_DOUBLE_EQ(*poly_system.val_p_spin[3 ], poly_graph.J(10, 11, 12)*spin[0]*spin[1]*spin[2]);
   EXPECT_DOUBLE_EQ(*poly_system.val_p_spin[4 ], poly_graph.J(10, 11)    *spin[0]*spin[1]        );
   EXPECT_DOUBLE_EQ(*poly_system.val_p_spin[5 ], poly_graph.J(10, 11, 12)*spin[0]*spin[1]*spin[2]);
   EXPECT_DOUBLE_EQ(*poly_system.val_p_spin[6 ], poly_graph.J(11, 12)    *spin[1]*spin[2]        );
   EXPECT_DOUBLE_EQ(*poly_system.val_p_spin[7 ], poly_graph.J(10, 11, 12)*spin[0]*spin[1]*spin[2]);
   EXPECT_DOUBLE_EQ(*poly_system.val_p_spin[8 ], poly_graph.J(10, 12)    *spin[0]*spin[2]        );
   EXPECT_DOUBLE_EQ(*poly_system.val_p_spin[9 ], poly_graph.J(10, 11, 12)*spin[0]*spin[1]*spin[2]);
   EXPECT_DOUBLE_EQ(*poly_system.val_p_spin[10], poly_graph.J(11, 12)    *spin[1]*spin[2]        );
   EXPECT_DOUBLE_EQ(*poly_system.val_p_spin[11], poly_graph.J(10, 11, 12)*spin[0]*spin[1]*spin[2]);
   
   //Check vartype
   EXPECT_TRUE(poly_system.isIsing);
   
   //Check variables for binary
   EXPECT_EQ(poly_system.val_binary.size(), 0);
   EXPECT_EQ(poly_system.zero_count_p_binary.size(), 0);
   EXPECT_EQ(poly_system.zero_count_binary.size(), 0);
   
}

TEST(PolySystem, ConstructorBinary) {
   
   openjij::graph::Index num_spins = 3;
   openjij::graph::Polynomial<double> poly_graph(num_spins, "BINARY");
   
   poly_graph.J(   {0}   ) = +0.0 ;//0
   poly_graph.J(   {1}   ) = +1.0 ;//1
   poly_graph.J(   {2}   ) = +2.0 ;//2
   poly_graph.J( {0, 1}  ) = +11.0;//3
   poly_graph.J( {0, 2}  ) = +22.0;//4
   poly_graph.J( {1, 2}  ) = +12.0;//5
   poly_graph.J({0, 1, 2}) = +12.0;//6
   
   openjij::graph::Spins spin = {1, 0, 1};
   
   openjij::system::ClassicalIsingPolynomial<openjij::graph::Polynomial<double>> poly_system = openjij::system::make_classical_ising_polynomial(spin, poly_graph);
   
   EXPECT_EQ(poly_system.num_spins, 3);
   for (auto i = 0; i < poly_system.num_spins; ++i) {
      EXPECT_EQ(poly_system.spin[i], spin[i]);
   }
   
   //Check J_term: set in SetJTerm()
   EXPECT_EQ(poly_system.J_term.size(), 7);
   EXPECT_DOUBLE_EQ(poly_system.J_term[0], 0 );
   EXPECT_DOUBLE_EQ(poly_system.J_term[1], 1 );
   EXPECT_DOUBLE_EQ(poly_system.J_term[2], 2 );
   EXPECT_DOUBLE_EQ(poly_system.J_term[3], 11);
   EXPECT_DOUBLE_EQ(poly_system.J_term[4], 22);
   EXPECT_DOUBLE_EQ(poly_system.J_term[5], 12);
   EXPECT_DOUBLE_EQ(poly_system.J_term[6], 12);
   
   //Check zero_count_binary: set in SetJTerm()
   EXPECT_EQ(poly_system.zero_count_binary.size(), 7);
   EXPECT_EQ(poly_system.zero_count_binary[0], 0);
   EXPECT_EQ(poly_system.zero_count_binary[1], 1);
   EXPECT_EQ(poly_system.zero_count_binary[2], 0);
   EXPECT_EQ(poly_system.zero_count_binary[3], 1);
   EXPECT_EQ(poly_system.zero_count_binary[4], 0);
   EXPECT_EQ(poly_system.zero_count_binary[5], 1);
   EXPECT_EQ(poly_system.zero_count_binary[6], 1);

   //Check connected_J_term_index: set in SetJTerm()
   EXPECT_EQ(poly_system.connected_J_term_index.size(), poly_system.num_spins);
   EXPECT_EQ(poly_system.connected_J_term_index[0].size(), 4);
   EXPECT_EQ(poly_system.connected_J_term_index[1].size(), 4);
   EXPECT_EQ(poly_system.connected_J_term_index[2].size(), 4);
   
   EXPECT_EQ(poly_system.connected_J_term_index[0][0], 0);
   EXPECT_EQ(poly_system.connected_J_term_index[0][1], 3);
   EXPECT_EQ(poly_system.connected_J_term_index[0][2], 4);
   EXPECT_EQ(poly_system.connected_J_term_index[0][3], 6);

   EXPECT_EQ(poly_system.connected_J_term_index[1][0], 1);
   EXPECT_EQ(poly_system.connected_J_term_index[1][1], 3);
   EXPECT_EQ(poly_system.connected_J_term_index[1][2], 5);
   EXPECT_EQ(poly_system.connected_J_term_index[1][3], 6);
   
   EXPECT_EQ(poly_system.connected_J_term_index[2][0], 2);
   EXPECT_EQ(poly_system.connected_J_term_index[2][1], 4);
   EXPECT_EQ(poly_system.connected_J_term_index[2][2], 5);
   EXPECT_EQ(poly_system.connected_J_term_index[2][3], 6);
   
   //Check dE: set in SetdE()
   EXPECT_DOUBLE_EQ(poly_system.dE.size(), poly_system.num_spins);
   EXPECT_DOUBLE_EQ(poly_system.dE[0], std::pow(-1, spin[0])*(poly_graph.J(0) + poly_graph.J(0, 1)*spin[1] + poly_graph.J(0, 2)*spin[2] + poly_graph.J(0, 1, 2)*spin[1]*spin[2]));
   EXPECT_DOUBLE_EQ(poly_system.dE[1], std::pow(-1, spin[1])*(poly_graph.J(1) + poly_graph.J(0, 1)*spin[0] + poly_graph.J(1, 2)*spin[2] + poly_graph.J(0, 1, 2)*spin[0]*spin[2]));
   EXPECT_DOUBLE_EQ(poly_system.dE[2], std::pow(-1, spin[2])*(poly_graph.J(2) + poly_graph.J(0, 2)*spin[0] + poly_graph.J(1, 2)*spin[1] + poly_graph.J(0, 1, 2)*spin[0]*spin[1]));

   //Check UpdateMatrix: set in SetUpdateMatrix()
   EXPECT_EQ(poly_system.row.size(), poly_system.num_spins + 1);
   EXPECT_EQ(poly_system.col.size(), 12);
   EXPECT_EQ(poly_system.val_binary.size(), 12);
   EXPECT_EQ(poly_system.zero_count_p_binary.size(), 12);

   EXPECT_EQ(poly_system.row[0 ], 0 );
   EXPECT_EQ(poly_system.row[1 ], 4 );
   EXPECT_EQ(poly_system.row[2 ], 8 );
   EXPECT_EQ(poly_system.row[3 ], 12);
   
   EXPECT_EQ(poly_system.col[0 ], 1);
   EXPECT_EQ(poly_system.col[1 ], 1);
   EXPECT_EQ(poly_system.col[2 ], 2);
   EXPECT_EQ(poly_system.col[3 ], 2);
   EXPECT_EQ(poly_system.col[4 ], 0);
   EXPECT_EQ(poly_system.col[5 ], 0);
   EXPECT_EQ(poly_system.col[6 ], 2);
   EXPECT_EQ(poly_system.col[7 ], 2);
   EXPECT_EQ(poly_system.col[8 ], 0);
   EXPECT_EQ(poly_system.col[9 ], 0);
   EXPECT_EQ(poly_system.col[10], 1);
   EXPECT_EQ(poly_system.col[11], 1);
   
   EXPECT_DOUBLE_EQ(poly_system.val_binary[0 ], poly_graph.J(0, 1)   );
   EXPECT_DOUBLE_EQ(poly_system.val_binary[1 ], poly_graph.J(0, 1, 2));
   EXPECT_DOUBLE_EQ(poly_system.val_binary[2 ], poly_graph.J(0, 2)   );
   EXPECT_DOUBLE_EQ(poly_system.val_binary[3 ], poly_graph.J(0, 1, 2));
   EXPECT_DOUBLE_EQ(poly_system.val_binary[4 ], poly_graph.J(0, 1)   );
   EXPECT_DOUBLE_EQ(poly_system.val_binary[5 ], poly_graph.J(0, 1, 2));
   EXPECT_DOUBLE_EQ(poly_system.val_binary[6 ], poly_graph.J(1, 2)   );
   EXPECT_DOUBLE_EQ(poly_system.val_binary[7 ], poly_graph.J(0, 1, 2));
   EXPECT_DOUBLE_EQ(poly_system.val_binary[8 ], poly_graph.J(0, 2)   );
   EXPECT_DOUBLE_EQ(poly_system.val_binary[9 ], poly_graph.J(0, 1, 2));
   EXPECT_DOUBLE_EQ(poly_system.val_binary[10], poly_graph.J(1, 2)   );
   EXPECT_DOUBLE_EQ(poly_system.val_binary[11], poly_graph.J(0, 1, 2));
   
   EXPECT_EQ(*poly_system.zero_count_p_binary[0 ], poly_system.zero_count_binary[3]);
   EXPECT_EQ(*poly_system.zero_count_p_binary[1 ], poly_system.zero_count_binary[6]);
   EXPECT_EQ(*poly_system.zero_count_p_binary[2 ], poly_system.zero_count_binary[4]);
   EXPECT_EQ(*poly_system.zero_count_p_binary[3 ], poly_system.zero_count_binary[6]);
   EXPECT_EQ(*poly_system.zero_count_p_binary[4 ], poly_system.zero_count_binary[3]);
   EXPECT_EQ(*poly_system.zero_count_p_binary[5 ], poly_system.zero_count_binary[6]);
   EXPECT_EQ(*poly_system.zero_count_p_binary[6 ], poly_system.zero_count_binary[5]);
   EXPECT_EQ(*poly_system.zero_count_p_binary[7 ], poly_system.zero_count_binary[6]);
   EXPECT_EQ(*poly_system.zero_count_p_binary[8 ], poly_system.zero_count_binary[4]);
   EXPECT_EQ(*poly_system.zero_count_p_binary[9 ], poly_system.zero_count_binary[6]);
   EXPECT_EQ(*poly_system.zero_count_p_binary[10], poly_system.zero_count_binary[5]);
   EXPECT_EQ(*poly_system.zero_count_p_binary[11], poly_system.zero_count_binary[6]);
   
   //Check vartype
   EXPECT_FALSE(poly_system.isIsing);
   
   //Check variables for binary
   EXPECT_EQ(poly_system.val_p_spin.size(), 0);
   
}

TEST(PolyUpdater, Quad) {
   
   //Check the polynomial updater work properly by comparing the result of the quadratic updater
   
   //generate classical sparse system
   const auto interaction = generate_interaction<openjij::graph::Sparse<double>>();
   auto engine_for_spin = std::mt19937(1);
   const auto spin = interaction.gen_spin(engine_for_spin);
   auto classical_ising = openjij::system::make_classical_ising(spin, interaction);
   
   auto random_numder_engine = std::mt19937(1);
   const auto schedule_list = generate_schedule_list();

   openjij::algorithm::Algorithm<openjij::updater::SingleSpinFlip>::run(classical_ising, random_numder_engine, schedule_list);
   
   //generate classical polynomial system
   const auto interaction_poly = generate_interaction<openjij::graph::Polynomial<double>>();
   auto engine_for_spin_poly = std::mt19937(1);
   const auto spin_poly = interaction.gen_spin(engine_for_spin_poly);
   auto classical_ising_poly = openjij::system::make_classical_ising_polynomial(spin_poly, interaction_poly);
   
   auto random_numder_engine_poly = std::mt19937(1);
   const auto schedule_list_poly = generate_schedule_list();
   
   openjij::algorithm::Algorithm<openjij::updater::SingleSpinFlip>::run(classical_ising_poly, random_numder_engine_poly, schedule_list_poly);
   
   EXPECT_EQ(openjij::result::get_solution(classical_ising_poly), openjij::result::get_solution(classical_ising));
}


