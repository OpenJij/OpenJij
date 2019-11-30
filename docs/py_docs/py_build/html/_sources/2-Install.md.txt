# Install

## CMake setup

OpenJij needs cmake >= 3.12.2

### macOS
Install newest CMake
```shell
$ brew install cmake
or
$ pip install cmake
```

### Linux (Ubuntu)

```shell
# if you installed old version by apt-get
$ apt-get purge cmake

# install cmake 
$ wget https://cmake.org/files/v3.13/cmake-3.13.2.tar.gz
$ tar xvf cmake-3.13.2.tar.gz
$ cd cmake-3.13.2
$ ./bootstrap && make && sudo make install 
```

## Install

### pip

```shell
$ pip install openjij
```

### from Github

```shell
$ git clone git@github.com:OpenJij/OpenJij.git
$ cd openjij
$ python setup.py install
```

## Note

### Windows enviroment
It has been reported that the installation of OpenJij may fail on Windows.
We recommend using Linux through WSL.

### GPU (CUDA)

If a CUDA compiler is found during installation, OpenJij will automatically compile for the GPU.
Compiling for GPU is planned to be optional, but it is not yet implemented (2019/11).


### Report an installation problem

Please report the problem to the issue on Github.

[https://github.com/OpenJij/OpenJij/issues](https://github.com/OpenJij/OpenJij/issues)