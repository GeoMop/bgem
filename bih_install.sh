#!/bin/bash

echo("Installing BIH locally.")
echo("g++ 4.x and cmake 3.x or newer are assumed.")

git submodule update --init --recursive
cd external/bih
mkdir build
cd build
cmake ..
make

BIH_PY_PATH=`pwd`
echo "Modifiing your .bashrc:"
echo "Add "$BIH_PY_PATH" to PYTHONPATH"
echo "PYTHONPATH=\"\$PYTHONPATH:$BIH_PY_PATH\"" 

