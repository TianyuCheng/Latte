#/bin/bash

make clean sbatch mini mnist-fc-fc SBATCH="./sbatches/mnist-none"
make clean sbatch mini tiling mnist-fc-fc SBATCH="./sbatches/mnist-t"
make clean sbatch mini tiling fusion mnist-fc-fc SBATCH="./sbatches/mnist-t-f"

make clean sbatch mini batch mnist-fc-fc SBATCH="./sbatches/mnist-b"
make clean sbatch mini batch tiling mnist-fc-fc SBATCH="./sbatches/mnist-b-t"
make clean sbatch mini batch tiling fusion mnist-fc-fc SBATCH="./sbatches/mnist-b-t-f"
