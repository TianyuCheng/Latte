CC		  := g++
CXX_FLAGS := -std=c++11 -fopenmp

BUILD_DIR := bin

all: bin CppLatte

bin:
	@mkdir -p $(BUILD_DIR)

CppLatte: Latte.cpp
	$(CC) $(CXX_FLAGS) $< -o $(BUILD_DIR)/$@

clean:
	rm -rf $(BUILD_DIR)
