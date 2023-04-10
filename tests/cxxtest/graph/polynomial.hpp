//    Copyright 2023 Jij Inc.

//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at

//        http://www.apache.org/licenses/LICENSE-2.0

//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.


#pragma once


namespace openjij {
namespace test {


TEST(PolyGraph, ConstructorCimodDenseInt) {
   TestPolyGraphConstructorCimodDense<openjij::graph::Index, double>(GeneratePolynomialInteractionsDenseInt<double>());
}
TEST(PolyGraph, ConstructorCimodDenseString) {
   TestPolyGraphConstructorCimodDense<std::string, double>(GeneratePolynomialInteractionsDenseString<double>());
}
TEST(PolyGraph, ConstructorCimodDenseTuple2) {
   TestPolyGraphConstructorCimodDense<std::tuple<openjij::graph::Index, openjij::graph::Index>, double>(GeneratePolynomialInteractionsDenseTuple2<double>());
}
TEST(PolyGraph, ConstructorCimodSparseInt1) {
   TestPolyGraphConstructorCimodSparse<openjij::graph::Index, double>(GeneratePolynomialInteractionsSparseInt1<double>());
}
TEST(PolyGraph, ConstructorCimodSparseInt2) {
   TestPolyGraphConstructorCimodSparse<openjij::graph::Index, double>(GeneratePolynomialInteractionsSparseInt2<double>());
}
TEST(PolyGraph, ConstructorCimodSparseInt3) {
   TestPolyGraphConstructorCimodSparse<openjij::graph::Index, double>(GeneratePolynomialInteractionsSparseInt3<double>());
}

TEST(PolyGraph, AddInteractions1) {
   const openjij::graph::Index num_spins = 3;
   openjij::graph::Polynomial<double> poly_graph(num_spins);
   
   poly_graph.J(   {}    ) = +0.1  ;
   poly_graph.J(   {0}   ) = -0.5  ;
   poly_graph.J(   {1}   ) = +1.0  ;
   poly_graph.J(   {2}   ) = -2.0  ;
   poly_graph.J( {0, 1}  ) = +10.0 ;
   poly_graph.J( {0, 2}  ) = -20.0 ;
   poly_graph.J( {1, 2}  ) = +21.0 ;
   poly_graph.J({0, 1, 2}) = -120.0;
   
   TestPolyGraphDense(poly_graph);
}

TEST(PolyGraph, AddInteractions2) {
   const openjij::graph::Index num_spins = 3;
   openjij::graph::Polynomial<double> poly_graph(num_spins);
   
   poly_graph.J(       ) = +0.1  /2.0;
   poly_graph.J(   0   ) = -0.5  /2.0;
   poly_graph.J(   1   ) = +1.0  /2.0;
   poly_graph.J(   2   ) = -2.0  /2.0;
   poly_graph.J( 0, 1  ) = +10.0 /2.0;
   poly_graph.J( 0, 2  ) = -20.0 /2.0;
   poly_graph.J( 1, 2  ) = +21.0 /2.0;
   poly_graph.J(0, 1, 2) = -120.0/2.0;
   
   poly_graph.J(       ) += +0.1  /2.0;
   poly_graph.J(   0   ) += -0.5  /2.0;
   poly_graph.J(   1   ) += +1.0  /2.0;
   poly_graph.J(   2   ) += -2.0  /2.0;
   poly_graph.J( 0, 1  ) += +10.0 /2.0;
   poly_graph.J( 0, 2  ) += -20.0 /2.0;
   poly_graph.J( 1, 2  ) += +21.0 /2.0;
   poly_graph.J(0, 1, 2) += -120.0/2.0;
   
   TestPolyGraphDense(poly_graph);
}

TEST(PolyGraph, AddInteractions3) {
   const openjij::graph::Index num_spins = 3;
   openjij::graph::Polynomial<double> poly_graph(num_spins);
   
   poly_graph.J(       ) = +0.1  ;
   poly_graph.J(   0   ) = -0.5  ;
   poly_graph.J(   1   ) = +1.0  ;
   poly_graph.J(   2   ) = -2.0  ;
   poly_graph.J( 0, 1  ) = +10.0 ;
   poly_graph.J( 0, 2  ) = -20.0 ;
   poly_graph.J( 1, 2  ) = +21.0 ;
   poly_graph.J(0, 1, 2) = -120.0;
   
   TestPolyGraphDense(poly_graph);
}

TEST(PolyGraph, AddInteractions4) {
   const openjij::graph::Index num_spins = 3;
   openjij::graph::Polynomial<double> poly_graph(num_spins);
   
   poly_graph.J(       ) = +999999;
   poly_graph.J(   0   ) = +999999;
   poly_graph.J(   1   ) = +999999;
   poly_graph.J(   2   ) = +999999;
   poly_graph.J( 0, 1  ) = +999999;
   poly_graph.J( 0, 2  ) = +999999;
   poly_graph.J( 1, 2  ) = +999999;
   poly_graph.J(0, 1, 2) = +999999;
   
   poly_graph.J(       ) = +0.1  ;
   poly_graph.J(   0   ) = -0.5  ;
   poly_graph.J(   1   ) = +1.0  ;
   poly_graph.J(   2   ) = -2.0  ;
   poly_graph.J( 0, 1  ) = +10.0 ;
   poly_graph.J( 0, 2  ) = -20.0 ;
   poly_graph.J( 1, 2  ) = +21.0 ;
   poly_graph.J(0, 1, 2) = -120.0;
   
   TestPolyGraphDense(poly_graph);
}




}
}
