CC		   := icpc -std=c++0x
CXX_FLAGS  := -DMKL_ILP64 -m64 -I${MKLROOT}/include -I${TACC_BOOST_INC} -openmp
LINK_FLAGS := -Wl,--start-group -L${TACC_BOOST_LIB} ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_sequential.a -Wl,--end-group -lpthread -lm -ldl -openmp

GEN		   := ./latte/generator.py
GEN_FLAGS  := 
NUM_WORKERS:= 5
tests      := fc-fc fc-conv-fc fc-meanp-fc

all: sbatch clean $(tests)

sbatch:
	@echo "#!/bin/bash"          > run.sbatch
	@echo "SBATCH -J latte"     >> run.sbatch
	@echo "SBATCH -o latte.out" >> run.sbatch
	@echo "SBATCH -e latte.err" >> run.sbatch
	@echo "SBATCH -n 2" 		   >> run.sbatch
	@echo "SBATCH -p normal"    >> run.sbatch
	@echo "SBATCH -t 00:00:20"  >> run.sbatch
	@echo ""  				   >> run.sbatch

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
	@echo "./$@" >> run.sbatch

%.cpp: test/%.py
	$(GEN) $(GEN_FLAGS) $< $@

mlp: mlp.cpp
	$(CC) -c ${CXX_FLAGS} $<
	$(CC) -o mlp mlp.o ${LINK_FLAGS}

clean:
	rm -rf $(BUILD_DIR)
	rm -rf *.o
	rm -rf $(tests)
