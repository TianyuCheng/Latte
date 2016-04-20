CC		   := g++ -std=c++11 
MKLROOT    := /home/jimmylin/intel/compilers_and_libraries/linux/mkl
CXX_FLAGS  := -DMKL_ILP64 -m64 -I${MKLROOT}/include 
LINK_FLAGS := -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_sequential.a -Wl,--end-group -lpthread -lm -ldl

BUILD_DIR := bin

all: bin CppLatte

bin:
	@mkdir -p $(BUILD_DIR)

CppLatte: Latte.cpp
	$(CC) -c ${CXX_FLAGS} $<
	$(CC) -o Latte Latte.o ${LINK_FLAGS}

mlp: mlp.cpp
	$(CC) -c ${CXX_FLAGS} $<
	$(CC) -o mlp mlp.o ${LINK_FLAGS}

clean:
	rm -rf $(BUILD_DIR)
	rm -rf *.o
	rm -rf Latte mlp

# $(CXX_FLAGS) $< -o $(BUILD_DIR)/$@
