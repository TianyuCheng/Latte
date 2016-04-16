CC		  := g++
CXX_FLAGS := -std=c++11

BUILD_DIR := bin

all: bin network ensemble neuron

bin:
	@mkdir -p $(BUILD_DIR)

neuron: Neuron.cpp
	$(CC) $(CXX_FLAGS) $< -o $(BUILD_DIR)/$@

ensemble: Ensemble.cpp
	$(CC) $(CXX_FLAGS) $< -o $(BUILD_DIR)/$@

network: Network.cpp
	$(CC) $(CXX_FLAGS) $< -o $(BUILD_DIR)/$@

clean:
	rm -rf $(BUILD_DIR)
