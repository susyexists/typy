#!/bin/bash 
#SBATCH -p bansil
#SBATCH -n 64
#SBATCH -N 1
#SBATCH -J typy
#SBATCH -o selfen.slurm
#SBATCH --time=7-00:00:00

module load anaconda3/3.7
python selfen.py
