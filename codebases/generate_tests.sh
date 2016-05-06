#/bin/bash

EXE="mnist-fc-fc"

make clean sbatch mini $EXE SBATCH="mnist-none"
mv $EXE $EXE-none
echo "./$EXE-none" >> ./mnist-none.sbatch

make clean sbatch mini tiling $EXE SBATCH="mnist-t"
mv $EXE $EXE-t
echo "./$EXE-t" >> ./mnist-t.sbatch

make clean sbatch mini tiling fusion $EXE SBATCH="mnist-t-f"
mv $EXE $EXE-t-f
echo "./$EXE-t-f" >> ./mnist-t-f.sbatch

make clean sbatch mini batch $EXE SBATCH="mnist-b"
mv $EXE $EXE-b
echo "./$EXE-b" >> ./mnist-b.sbatch

make clean sbatch mini batch tiling $EXE SBATCH="mnist-b-t"
mv $EXE $EXE-b-t
echo "./$EXE-b-t" >> ./mnist-b-t.sbatch

make clean sbatch mini batch tiling fusion $EXE SBATCH="mnist-b-t-f"
mv $EXE $EXE-b-t-f
echo "./$EXE-b-t-f" >> ./mnist-b-t-f.sbatch
