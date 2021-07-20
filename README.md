# OpenJij : Framework for the Ising model and QUBO.

![build-test](https://github.com/OpenJij/OpenJij/workflows/build-test/badge.svg)

* python >= 3.7
* (optional) gcc >= 7.0.0
* (optional) cmake >= 3.17
* (optional) Ninja 

- [Documents](https://openjij.github.io/OpenJij_Documentation/build/html/)
- [Tutorials](https://openjij.github.io/OpenJijTutorial/build/html/index.html)
- [C++ Docs](https://openjij.github.io/OpenJij/html/index.html)

## install

### install via pip
> Note: To use GPGPU algorithms, please follow the section [`install via pip from source codes`](#install-via-pip-from-source-codes) below.
> GPGPU algorithms are automatically enabled once CMake finds CUDA frameworks during installation.
```
$ pip install openjij
```

### install via pip from source codes
To install OpenJij from source codes, please install CMake first then install OpenJij.

#### cmake setup
If you want to use setup.py instead of PIP, You will need to install CMake\>\=3.17.  
We are Highly recommended install CMake via PYPI.
```
$ pip install -U cmake
```
* macOS
```
$ brew install cmake
```

* Linux
```
# if you installed old version by apt-get
$ apt-get purge cmake

# install cmake 
$ wget https://cmake.org/files/v3.17/cmake-3.17.5.tar.gz
$ tar xvf cmake-3.17.5.tar.gz
$ cd cmake-3.17.5
$ ./bootstrap && make && sudo make install 
```

* Windows

Please install cmake from [here](https://cmake.org/download/).


Make sure the enviroment path for CMake is set correctly.

#### install OpenJij
```
$ pip install openjij --no-binary :all:
```

### install from github repository
```
$ git clone git@github.com:OpenJij/OpenJij.git
$ cd openjij
$ python -m pip install .
```

## Test

### Test only Python code
```sh
$ python setup.py test
```

### Test Python and C++ code
```sh
$ export USE_TEST=1
$ python setup.py test
```

## How to use

### Python example

```python
import openjij as oj
sampler = oj.SASampler()
response = sampler.sample_ising(h={0: -1}, J={(0,1): -1})
response.states
# [[1,1]]

# with indices
response = sampler.sample_ising(h={'a': -1}, J={('a','b'): 1})
[{index: s for index, s in zip(response.indices, state)} for state in response.states]
# [{'b': -1, 'a': 1}]
```

### C++ example

```
$ source ./build_gcc.sh
$ cd ./project_template
$ make
$ ./tutorial
The result spins are [1 1 1 1 1 ]
```

## Community

- [OpenJij Slack](https://join.slack.com/t/openjij/shared_invite/enQtNjQyMjIwMzMwNzA4LTQ5MWRjOWYxYmY1Nzk4YzdiYzlmZjIxYjhhMmMxZjAyMzE3MDc1ZWRkYmI1YjhkNjRlOTM1ODE0NTc5Yzk3ZDA)

## About us

This product is maintained by Jij Inc.

**Please visit our website for more information!**
https://j-ij.com/

