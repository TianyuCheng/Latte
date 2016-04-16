CC		  := g++
CXX_FLAGS := -std=c++11

all: network ensemble neuron

neuron: Neuron.cpp
	$(CC) $(CXX_FLAGS) $< -o $@

ensemble: Ensemble.cpp
	$(CC) $(CXX_FLAGS) $< -o $@

network: Network.cpp
	$(CC) $(CXX_FLAGS) $< -o $@

clean:
	rm -rf network ensemble neuron
