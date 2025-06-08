#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --time=1:30:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=try_llamav_eval_sqa
#SBATCH --output=llamav-sqa-eval-%j-%a.out # set the output file name for each task. The %j is replaced by the job ID and %a is replaced by the task ID.


module purge
cd /scratch/yw6594/cf/vlm/LLaVA
source /vast/yw6594/miniforge3/bin/activate vlmm
bash /scratch/yw6594/cf/vlm/LLaVA/playground/data/sqa.sh

# cd  /scratch/yw6594/cf/vlm/LLaVA/playground/llamav_out