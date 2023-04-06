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

TEST(PolySystemKLP, ConstructorCimodDenseInt) {
   TestKLPConstructorCimod<openjij::graph::Index, double>(GeneratePolynomialInteractionsDenseInt<double>(), "Dense");
}
TEST(PolySystemKLP, ConstructorCimodDenseString) {
   TestKLPConstructorCimod<std::string, double>(GeneratePolynomialInteractionsDenseString<double>(), "Dense");
}
TEST(PolySystemKLP, ConstructorCimodDenseTuple2) {
   TestKLPConstructorCimod<std::tuple<openjij::graph::Index, openjij::graph::Index>, double>(GeneratePolynomialInteractionsDenseTuple2<double>(), "Dense");
}
TEST(PolySystemKLP, ConstructorCimodSparseInt1) {
   TestKLPConstructorCimod<openjij::graph::Index, double>(GeneratePolynomialInteractionsSparseInt1<double>(), "Sparse");
}
TEST(PolySystemKLP, ConstructorCimodSparseInt2) {
   TestKLPConstructorCimod<openjij::graph::Index, double>(GeneratePolynomialInteractionsSparseInt2<double>(), "Sparse");
}
TEST(PolySystemKLP, ConstructorCimodSparseInt3) {
   TestKLPConstructorCimod<openjij::graph::Index, double>(GeneratePolynomialInteractionsSparseInt3<double>(), "Sparse");
}
TEST(PolySystemKLP, ConstructorGraphDenseInt) {
   TestKLPConstructorGraph<openjij::graph::Index, double>(GeneratePolynomialInteractionsDenseInt<double>(), "Dense");
}
TEST(PolySystemKLP, ConstructorGraphSparseInt1) {
   TestKLPConstructorGraph<openjij::graph::Index, double>(GeneratePolynomialInteractionsSparseInt1<double>(), "Sparse");
}
TEST(PolySystemKLP, ConstructorGraphSparseInt2) {
   TestKLPConstructorGraph<openjij::graph::Index, double>(GeneratePolynomialInteractionsSparseInt2<double>(), "Sparse");
}

}
}
