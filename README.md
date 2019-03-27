# OpenJij : Framework for the Ising model and QUBO.

[![Build Status](https://travis-ci.org/OpenJij/OpenJij.svg?branch=master)](https://travis-ci.org/OpenJij/OpenJij)

cmake >= 3.11

Documents : https://openjij.github.io/OpenJij/

## install

### pip
```
$ pip install OpenJij
```

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

### Install openjij 
```
$ git clone git@github.com:OpenJij/OpenJij.git
$ cd openjij
$ python setup.py install
```

## How to use

### Python example

#### for simulated / simulated quantum annealing

```python
# import modules
import cxxjij.graph as G #graphs
import cxxjij.method as M #methods
J = G.Dense(10) #Dense graph with 10 spins

for i in range(10):
	for j in range(10):
		J[i, j] = -0.1 if i is not j else 0 #ferrromagnetic interaction

# Simulated annealing (classical)
cising = M.ClassicalIsing(J)
cising.simulated_annealing(0.01, 10, 100, 10) 
print("SA results: ", cising.get_spins())


# Simulated quantum annealing (quantum simulation) with 10 trotter slices
qising = M.QuantumIsing(J, 10)
qising.simulated_annealing(10, 0.01, 10, 100, 10) 
print("SQA results: ", qising.get_spins())
```
