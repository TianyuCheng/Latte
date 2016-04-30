#!/bin/bash
#SBATCH -J mlp            # job name
#SBATCH -o omlp.out
#SBATCH -e emlp.out
#SBATCH -n 2              # total number of mpi tasks requested
#SBATCH -p normal     
#SBATCH -t 00:00:10        

./mlp
