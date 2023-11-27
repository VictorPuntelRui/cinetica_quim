#!/bin/bash
#SBATCH -n 4
#SBATCH --ntasks-per-node=4
#SBATCH -p head
#SBATCH --mem=8000mb
#SBATCH --time=30:00:00

source ~/.bashrc
conda activate ilumpy

python3 ver6_19_6_no_UI.py

#jupyter-notebook --ip=172.20.10.15

