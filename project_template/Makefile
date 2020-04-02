#Specify the location of include files of OpenJij. 
INCLUDES=-I../src 

#Eigen directory
EIGEN_DIR=-I../build/_deps/eigen-src
#nlohmann_json directory
JSON_DIR=-I../build/_deps/nlohmann_json-src/single_include

# add GPU static library path here if needed
#LIBPATH=-L../build/src/system -lcxxjij_chimera_gpu_kernel_static

#add your path to cuda include files if you use cuda
#CUDA_DIR=-I/opt/cuda/include

# add GPU static library path here if needed
#LIBPATH=-L../build/src/system -lcxxjij_chimera_gpu_kernel_static

CXX=g++
CC=gcc
CFLAGS=-Wall 
CXXFLAGS=-Wextra -std=c++17 -Wall -O2 

# if you use CUDA, define USE_CUDA macro
#CPPFLAGS=-DUSE_CUDA

#program name
PROG=tutorial

.PHONY: all clean
all: $(PROG) 
	@echo Make Complete!
$(PROG): template.cpp
	$(CXX) template.cpp $(INCLUDES) $(EIGEN_DIR) $(JSON_DIR) $(CUDA_DIR) $(CXXFLAGS) $(CPPFLAGS) $(LIBPATH) -o $(PROG)
clean:
	$(RM) $(PROG)
