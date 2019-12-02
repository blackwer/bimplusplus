CXX ?= g++

bimplusplus: src/bimplusplus.cpp src/function_generator.hpp
	$(CXX) src/bimplusplus.cpp -o bimplusplus -I${CONDA_PREFIX}/include -I${CONDA_PREFIX}/include/eigen3 -g -O3 -march=native -ffast-math -DNDEBUG -std=c++17 -Wall -L${CONDA_PREFIX}/lib -Wl,-rpath ${CONDA_PREFIX}/lib -lhdf5 -lgsl

clean:
	rm -f bimplusplus

