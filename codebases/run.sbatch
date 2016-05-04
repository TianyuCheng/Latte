#!/bin/bash
SBATCH -J latte
SBATCH -o latte.out
SBATCH -e latte.err
SBATCH -n 2
SBATCH -p normal
SBATCH -t 00:00:20

./fc-conv-fc
./fc-meanp-fc
./fc-relu
./iris-fc-conv-fc
./iris-fc-fc
./iris-fc-meanp-fc
./iris-fc-relu
./mnist-fc-conv-fc
./mnist-fc-fc
./mnist-fc-meanp-fc
