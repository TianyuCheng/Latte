CC		   := icpc -std=c++0x
CXX_FLAGS  := -DMKL_ILP64 -m64 -I${MKLROOT}/include -I${TACC_BOOST_INC} -openmp
LINK_FLAGS := -Wl,--start-group -L${TACC_BOOST_LIB} ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_sequential.a -Wl,--end-group -lpthread -lm -ldl -openmp

GEN		   := ./latte/generator.py
GEN_FLAGS  := 
NUM_WORKERS:= 5
tests      := fc-fc fc-conv-fc fc-meanp-fc

all: clean $(tests)

mkl:
	GEN_FLAGS := $(GEN_FLAGS) -m

batch:
	GEN_FLAGS := $(GEN_FLAGS) -b -w $(NUM_WORKERS)

tiling:
	GEN_FLAGS := $(GEN_FLAGS) -t

fusion:
	GEN_FLAGS := $(GEN_FLAGS) -f

%: %.cpp
	$(CC) -c ${CXX_FLAGS} $< -o $<.o
	$(CC) -o $@ $<.o ${LINK_FLAGS}

%.cpp: test/%.py
	$(GEN) $(GEN_FLAGS) $< $@

mlp: mlp.cpp
	$(CC) -c ${CXX_FLAGS} $<
	$(CC) -o mlp mlp.o ${LINK_FLAGS}

clean:
	rm -rf $(BUILD_DIR)
	rm -rf *.o
	rm -rf $(tests)
