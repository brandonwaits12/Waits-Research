#!/bin/bash

#SBATCH --time=16:00:00   # walltime
#SBATCH --ntasks=32   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=4096M   # memory per CPU core
#SBATCH --mail-user=bwaits@byu.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
cd /home/bwaits/Research/Waits-Research/labs
source /home/bwaits/Research/Waits-Research/.venv/bin/activate
python illiquidity_main.py