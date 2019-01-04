# OpenJij : Framework for Ising model and QUBO.

cmake >= 3.11

## install

### cmake setup

If you had not installed cmake>=3.11, you need install cmake>=3.11

macOS
```
$ brew install cmake
```

Linux
```
# if you installed old version by apt-get
$ apt-get purge cmake

# install cmake 
$ wget https://cmake.org/files/v3.13/cmake-3.13.2.tar.gz
$ tar xvf cmake-3.13.2.tar.gz
$ cd cmake-3.13.2
$ ./bootstrap && make && sudo make install 
```

#### install openjij 
```
$ git clone git@github.com:OpenJij/OpenJij.git
$ cd openjij
$ python setup.py install
```

## How to use

### Python example

#### for optimizaiton

```python
import numpy as np
import openjij as oj
h = {0: -1}
J = {(0, 1): -1, (1, 2): -1}

bqm = oj.BinaryQuadraticModel(h, J) 
samp = oj.Sampler(bqm)

# Simulated annealing (classical) 10 times
response = samp.simulated_annealing(iteration=10)
# show the lowest energy solution in ten times
min_index = np.argmin(response.energies)
print("SA results: ", response.states[min_index])
# > SA results:  [1, 1, 1]


# Simulated quantum annealing (quantum simulation) 10 times
response = samp.simulated_quantum_annealing(iteration=10)
# show the lowest energy solution in ten times
min_index = np.argmin(response.energies)
print("SQA results: ", response.states[min_index])
# > SQA results:  [1, 1, 1]
```


### C++ example
see full example code at ``examples/cpp/example_optimization.cpp``

1. make "interaction matrix" and instances of "Sampler" and "Results"
```c++
// make "interaction matrix (class is SquareMatrix)" 
// which should be symmetry matrix.
// In the follow case, make 2D ferro magnetic interacaion.
const int L=20;
int N=L*L;
openjij::SquareMatrix<double> int_mat{N, 0.0};
for(int x=0; x < L; x++){
    for(int y=0; y < L; y++){
        int pos = x + y*L;
        int n_pos = (x+1)%L + y*L;
        int d_pos = x + (y+1)%L * L;
        int_mat(pos, n_pos) = -1.0;
        int_mat(pos, d_pos) = -1.0;
        int_mat(n_pos, pos) = -1.0;
        int_mat(d_pos, pos) = -1.0;
    }
}

// create instance of Sampler class
openjij::sampler::Sampler samp(int_mat);

// make Results instance which stores results of sampling or optimization.
openjij::sampler::Results resu;
```

next, if you want to do sampling go to "For sampling", otherwise if you want to do optimization go to "For optimization".

### For sampling
See full code "openjij/test/exmaple2d.cpp".
```c++
// paramters
int burn_out = 1000;  // thermalization
int observe_num = 10; // number of observation
double beta = 0.1;
```
if you want to do sampling from classical thermal equilibrium state, use .sampling().
``` c++
// classical sampling ----------------------------------
// (last argument is Results class for to store results)
samp.sampling(beta, burn_out, observe_num, resu);

// results are stored to "resu.states (std::vector<Spins>)"
```
``` c++
// quantum sampling ----------------------------------
double gamma = 2.0;
int trotter = 3;
samp.quantum_sampling(beta, gamma, burn_out, observe_num, trotter, resu);

// results are stored to "resu.quantum_states (std::vector<std::vector<Spins>>)"
```

### For optimization
```c++
// parameters
int step_length = 10;
int step_num = 10;
int iter = 10;
```

``` c++
// simulated annealing ----------------------------------
double beta_min = 0.6;
double beta_max = 2.0;
samp.simulated_annealing(beta_min, beta_max, step_length, step_num, iter, resu);

// results are stored to "resu.states"
```

``` c++
// simulated quantum annealing ----------------------------------
int trotter = 3;
double beta = 1.0;
double gamma_min = 1.0;
double gamma_max = 5.0;
samp.simulated_quantum_annealing(beta, gamma_min, gamma_max, trotter, step_length, step_num, iter, resu);

// results are stored to "resu.quantum_states"
```

