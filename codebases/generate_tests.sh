#/bin/bash

EXE="mnist-fc-fc"

function generate() { 
    make clean sbatch mini $FLAGS $EXE SBATCH=${EXE}${SUFFIX}
    mv $EXE ${EXE}${SUFFIX}
    echo ./${EXE}${SUFFIX} >> ./${EXE}${SUFFIX}.sbatch
    sbatch ./${EXE}${SUFFIX}.sbatch
} 

FLAGS=""
SUFFIX="-none"
generate

FLAGS="tiling"
SUFFIX="-t"
generate

# FLAGS="tiling fusion"
# SUFFIX="-t-f"
# generate

FLAGS="batch"
SUFFIX="-b"
generate

FLAGS="batch tiling"
SUFFIX="-b-t"
generate

# FLAGS="batch tiling fusion"
# SUFFIX="-b-t-f"
# generate

FLAGS="mkl"
SUFFIX="-m"
generate

FLAGS="mkl tiling"
SUFFIX="-m-t"
generate

# FLAGS="mkl tiling fusion"
# SUFFIX="-m-t-f"
# generate

FLAGS="mkl batch"
SUFFIX="-m-b"
generate

FLAGS="mkl batch tiling"
SUFFIX="-m-b-t"
generate

# FLAGS="mkl batch tiling fusion"
# SUFFIX="-m-b-t-f"
# generate
