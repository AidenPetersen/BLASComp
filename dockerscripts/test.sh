cmake . -B build
cd build
make
cd myblas/level1
ctest
cd ../level2
ctest
cd ../level3
ctest