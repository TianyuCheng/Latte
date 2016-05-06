#CC		   := icpc -std=c++0x
#CXX_FLAGS  := -DMKL_ILP64 -m64 -I${MKLROOT}/include -I${TACC_BOOST_INC} -openmp -lrt
#LINK_FLAGS := -Wl,--start-group -L${TACC_BOOST_LIB} ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_sequential.a -Wl,--end-group -lpthread -lm -ldl -openmp -lrt

CC		   := g++ -std=c++11
MKLROOT    := /home/jimmylin/intel/compilers_and_libraries/linux/mkl
CXX_FLAGS  := -DMKL_ILP64 -m64 -I${MKLROOT}/include -fopenmp -lrt
LINK_FLAGS := -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_sequential.a -Wl,--end-group -lpthread -lm -ldl -fopenmp -lrt

GEN		   := ./latte/generator.py
GEN_FLAGS  := 
NUM_WORKERS:= 5
test_dir   := ../testcases
tests      := $(patsubst %.py, %, $(shell (cd $(test_dir); ls *.py)))
deletes    := $(patsubst %, %.cpp, $(tests))

all: sbatch clean $(tests)

Latte: Latte.cpp
	$(CC) -c ${CXX_FLAGS} $<
	$(CC) -o Latte Latte.o ${LINK_FLAGS}

sbatch:
	@echo "#!/bin/bash"          > run.sbatch
	@echo "#SBATCH -J latte"     >> run.sbatch
	@echo "#SBATCH -o latte.out" >> run.sbatch
	@echo "#SBATCH -e latte.err" >> run.sbatch
	@echo "#SBATCH -n 2" 	     >> run.sbatch
	@echo "#SBATCH -p normal"    >> run.sbatch
	@echo "#SBATCH -t 00:00:20"  >> run.sbatch
	@echo ""  				     >> run.sbatch

mkl:
	$(eval GEN_FLAGS := $(GEN_FLAGS) -m)

batch:
	$(eval GEN_FLAGS := $(GEN_FLAGS) -b -w $(NUM_WORKERS))

tiling:
	$(eval GEN_FLAGS := $(GEN_FLAGS) -t)

fusion:
	$(eval GEN_FLAGS := $(GEN_FLAGS) -f)

mini:
	$(eval GEN_FLAGS := $(GEN_FLAGS) --mini)

%: %.cpp
	$(CC) -c ${CXX_FLAGS} $< -o $<.o
	$(CC) -o $@ $<.o ${LINK_FLAGS}
	@echo "./$@" >> run.sbatch

%.cpp: $(test_dir)/%.py
	$(GEN) $(GEN_FLAGS) $< $@

mfc: mnist-fc-fc.cpp
	$(CC) -c ${CXX_FLAGS} $<
	$(CC) -o mnist-fc-fc mnist-fc-fc.o ${LINK_FLAGS}

clean:
	rm -rf $(BUILD_DIR)
	rm -rf *.o
	rm -rf $(tests)
	rm -rf $(deletes)
	rm -rf latte.out latte.err

.PRECIOUS: %.cpp
