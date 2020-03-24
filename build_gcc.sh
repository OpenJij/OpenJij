#!/usr/bin/bash

BUILD_TYPE=Release

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=$BUILD_TYPE
cmake --build . --target python --config $BUILD_TYPE
cd ..
