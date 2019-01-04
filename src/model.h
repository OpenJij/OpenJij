//    Copyright 2019 Jij Inc.

//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at

//        http://www.apache.org/licenses/LICENSE-2.0

//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.


#ifndef MODEL
#define MODEL
#include <vector>
namespace openjij{
    using Spins = std::vector<int>;
    template <typename num> 
    struct SquareMatrix{
        public:
        SquareMatrix(int L, num value): L(L){data_vector.resize(L*L, value);}
        num& operator()(int x, int y){
            // #ifdef DEBUG
                if (x < 0 || x >= L || y < 0 || y >= L) throw "but access\n.";
            // #endif // ifdef DEBUG
            int n = x + y * L;
            return data_vector[n];
        }

        const num& operator()(int x, int y) const {
            // #ifdef DEBUG
                if (x < 0 || x >= L || y < 0 || y >= L) throw "but access\n.";
            // #endif // ifdef DEBUG
            int n = x + y * L;
            return data_vector[n];
        }
	// setter for python
	void set(int x, int y, num& value){
	    data_vector[x + y * L] = value;
	} 
        const int size() const {return L;}
        private:
            int L;
            std::vector<num> data_vector; 
    };
}

#endif // MODEL
