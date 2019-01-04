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

#ifndef UPDATER
#define UPDATER

#include "../model.h"

#include <vector>
#include <iostream>

namespace openjij{
    namespace updater{
        struct Updater{
            Updater(const SquareMatrix<double>& int_mat): N(int_mat.size()), int_mat(int_mat){
                adj_mat.resize(N, std::vector<int>());
                for(int i=0; i < N-1; i++){
                    for (int j=i+1; j < N; j++){
                        if(int_mat(i, j) != 0){
                            adj_mat[j].push_back(i);
                            adj_mat[i].push_back(j); 
                        }
                    }
                }
            }
			virtual ~Updater(){}
            virtual void spins_update(Spins& spins, double beta) = 0;
            virtual void quantum_spins_update(std::vector<Spins>& trotterized_spins, double beta, double gamma) = 0;
            #ifdef DEBUGTEST
            public:
            #else
            protected:
            #endif // DEBUGTEST
                const int N;
                const SquareMatrix<double>& int_mat;
                std::vector<std::vector<int>> adj_mat;
        };
    }
}

#endif // UPDATER
