CC		   := icpc -std=c++0x
# CC		   := g++ -std=c++11 
# TACC_BOOST_INC=/opt/apps/intel15/boost/1.55.0/x86_64/include
# TACC_BOOST_LIB=/opt/apps/intel15/boost/1.55.0/x86_64/lib
# MKLROOT    := /home/jimmylin/intel/compilers_and_libraries/linux/mkl
CXX_FLAGS  := -DMKL_ILP64 -m64 -I${MKLROOT}/include -I${TACC_BOOST_INC} -openmp
LINK_FLAGS := -Wl,--start-group -L${TACC_BOOST_LIB} ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_sequential.a -Wl,--end-group -lpthread -lm -ldl -openmp


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
