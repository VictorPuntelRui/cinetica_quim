#!/bin/bash
#SBATCH -n 8
#SBATCH --ntasks-per-node=8
#SBATCH -p head
#SBATCH --mem=35000mb
#SBATCH --time=60:00:00

source ~/.bashrc
conda activate ilumpy

python3 ver6_15_no_UI_py.py

#jupyter-notebook --ip=172.20.10.15

