#!/bin/bash
#
#SBATCH --job-name=test
#SBATCH --partition=deep
#SBATCH --time=20:00
#SBATCH --ntasks=1
source /sailhome/behzad/cs379c-project/gym/roper/bin/activate
python train.py --num_trials 1 --method PER --variance 10 --mean 0 --decision_eps 1 --hardcoded False
