# OpenJij : Framework for the Ising model and QUBO.

[![Build Status](https://travis-ci.org/OpenJij/OpenJij.svg?branch=master)](https://travis-ci.org/OpenJij/OpenJij)

cmake >= 3.11

Documents : https://openjij.github.io/OpenJij/

## install

> Notice：  
> Because current version v.0.0.1 support only the interface of 'cxxjij modules', you have to read header files in `src/*`

### pip
```
$ pip install openjij
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

```python
from openjij import graph as G   # or import cxxjij.graph as G
from openjij import method as M  # or import cxxjij.method as M
ising = G.Dense(10)
for i in range(10):
    for j in range(i+1, 10):
            ising[i, j] = -1

sa = M.ClassicalIsing(ising)
sa.simulated_annealing(0.01, 10, 100, 10)  # beta_min, beta_max, step_length, step_num
sa.get_spins()
>> [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
```
