# OpenJij : Framework for the Ising model and QUBO.

![build-test](https://github.com/OpenJij/OpenJij/workflows/build-test/badge.svg)

* python >= 3.7
* (optional) gcc >= 7.0.0
* (optional) cmake >= 3.12.2

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

If you had not installed cmake>=3.12.2, you need install cmake>=3.12.2

* macOS
```
$ brew install cmake
```

* Linux
```
# if you installed old version by apt-get
$ apt-get purge cmake

# install cmake 
$ wget https://cmake.org/files/v3.13/cmake-3.13.2.tar.gz
$ tar xvf cmake-3.13.2.tar.gz
$ cd cmake-3.13.2
$ ./bootstrap && make && sudo make install 
```

* Windows

Please install cmake from [here](https://cmake.org/download/).

Alternatively, you can use
```
$ pip install -U cmake
```
Make sure the enviroment path for CMake is set correctly.

#### install OpenJij
```
$ pip install openjij --no-binary :all:
```

### install from github repository
```
$ git clone git@github.com:OpenJij/OpenJij.git
$ cd openjij
$ python setup.py install
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

