#/bin/bash

EXE="mnist-fc-fc"

mkdir -p sbatch

make clean sbatch mini $EXE SBATCH="./sbatch/mnist-none"
mv $EXE ./sbatch/$EXE-none
echo "./$EXE-none" >> ./sbatch/mnist-none.sbatch

make clean sbatch mini tiling $EXE SBATCH="./sbatch/mnist-t"
mv $EXE ./sbatch/$EXE-t
echo "./$EXE-t" >> ./sbatch/mnist-t.sbatch

make clean sbatch mini tiling fusion $EXE SBATCH="./sbatch/mnist-t-f"
mv $EXE ./sbatch/$EXE-t-f
echo "./$EXE-t-f" >> ./sbatch/mnist-t-f.sbatch

make clean sbatch mini batch $EXE SBATCH="./sbatch/mnist-b"
mv $EXE ./sbatch/$EXE-b
echo "./$EXE-b" >> ./sbatch/mnist-b.sbatch

make clean sbatch mini batch tiling $EXE SBATCH="./sbatch/mnist-b-t"
mv $EXE ./sbatch/$EXE-b-t
echo "./$EXE-b-t" >> ./sbatch/mnist-b-t.sbatch

make clean sbatch mini batch tiling fusion $EXE SBATCH="./sbatch/mnist-b-t-f"
mv $EXE ./sbatch/$EXE-b-t-f
echo "./$EXE-b-t-f" >> ./sbatch/mnist-b-t-f.sbatch
