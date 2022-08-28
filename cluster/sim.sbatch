#!/bin/bash
#
###############################################################################
# sim.sbatch
# 
# Batch script that runs some number of trials in the simulation and writes out 
# the output to disc.
###############################################################################
#                                                                            
#SBATCH --time=24:00:00                  # Job run time (hh:mm:ss)
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks-per-node=16             # Number of task (cores/ppn) per node
#SBATCH --job-name=potential             # Name of batch job
#SBATCH --partition=csl                  # Partition (queue)
##SBATCH --array=1                        # Job array indicies         
#SBATCH --output=jobarray.o%A_%a         # Name of batch job output file
#SBATCH --error=jobarray.e%A_%a          # Name of batch job error file
#SBATCH --mail-user=zjw4@illinois.edu    # Send email notifications
##SBATCH --mail-type=END                  # Type of email notifications to send
#                                                                            
###############################################################################

# Create a unique directory using the jobid
mkdir -p ${SLURM_SUBMIT_DIR}/${SLURM_ARRAY_TASK_ID}
cd ${SLURM_SUBMIT_DIR}/${SLURM_ARRAY_TASK_ID}

module load anaconda/2021-May/3
conda activate dec

# Q: why is this necessary?
conda init bash
exec $SHELL

python ../run/analysis.py


