CC		   := icpc -std=c++0x
CXX_FLAGS  := -O3 -DMKL_ILP64 -m64 -I${MKLROOT}/include -I${TACC_BOOST_INC} -openmp -lrt
LINK_FLAGS := -O3 -Wl,--start-group -L${TACC_BOOST_LIB} ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_sequential.a -Wl,--end-group -lpthread -lm -ldl -openmp -lrt

# CC		   := g++ -std=c++11
# MKLROOT    := /home/jimmylin/intel/compilers_and_libraries/linux/mkl
# CXX_FLAGS  := -O3 -DMKL_ILP64 -m64 -I${MKLROOT}/include -fopenmp -lrt
# LINK_FLAGS := -O3 -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_sequential.a -Wl,--end-group -lpthread -lm -ldl -fopenmp -lrt

GEN		   := ./latte/generator.py
GEN_FLAGS  := 
NUM_WORKERS:= 5
test_dir   := ../testcases
tests      := $(patsubst %.py, %, $(shell (cd $(test_dir); ls *.py)))
deletes    := $(patsubst %, %.cpp, $(tests))

SBATCH	   := run.sbatch

all: clean $(tests)

Latte: Latte.cpp
	$(CC) -c ${CXX_FLAGS} $<
	$(CC) -o Latte Latte.o ${LINK_FLAGS}

sbatch:
	@echo "#!/bin/bash"              >  $(SBATCH).sbatch
	@echo "#SBATCH -J $(SBATCH)"     >> $(SBATCH).sbatch 
	@echo "#SBATCH -o $(SBATCH).out" >> $(SBATCH).sbatch 
	@echo "#SBATCH -e $(SBATCH).err" >> $(SBATCH).sbatch 
	@echo "#SBATCH -n 2" 	     	 >> $(SBATCH).sbatch 
	@echo "#SBATCH -p normal"    	 >> $(SBATCH).sbatch 
	@echo "#SBATCH -t 00:15:00"  	 >> $(SBATCH).sbatch 
	@echo ""  				     	 >> $(SBATCH).sbatch 

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

%.cpp: $(test_dir)/%.py
	$(GEN) $(GEN_FLAGS) $< $@

mfc: mnist-fc-fc.cpp
	$(CC) -c ${CXX_FLAGS} $<
	$(CC) -o mnist-fc-fc mnist-fc-fc.o ${LINK_FLAGS}

clean_sbatch:
	rm -rf *.sbatch
	rm -rf *.out
	rm -rf *.err

clean:
	rm -rf $(BUILD_DIR)
	rm -rf *.o
	rm -rf $(tests)
	rm -rf $(deletes)
	rm -rf latte.out latte.err

.PHONY: batch fusion tiling mini sbatch
.PRECIOUS: %.cpp
