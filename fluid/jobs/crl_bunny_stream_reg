#!/bin/bash

#SBATCH -n 16                              # Number of cores
#SBATCH --time=20:00:00                      # hours:minutes:seconds
#SBATCH --mem-per-cpu=2000
#SBATCH --tmp=4000                        # per node!!
#SBATCH --job-name=crl_bunny_stream_reg
#SBATCH --output=./jobs/crl_bunny_stream_reg.out
#SBATCH --error=./jobs/crl_bunny_stream_reg.err

#env2lmod
module load gcc/8.2.0 python/3.9.9 cmake/3.25.0 freeglut/3.0.0 libxrandr/1.5.0  libxinerama/1.1.3 libxi/1.7.6  libxcursor/1.1.14 mesa/17.2.3 eth_proxy
# run experiment
conda activate pbs 
python3 crl_bunny_stream_reg.py
