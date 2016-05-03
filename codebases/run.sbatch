#!/bin/bash
SBATCH -J latte
SBATCH -o latte.out
SBATCH -e latte.err
SBATCH -n 2
SBATCH -p normal
SBATCH -t 00:00:20

