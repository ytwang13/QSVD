#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --time=1:30:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-7 # here replace with other interation
#SBATCH --job-name=MMCL_KDranks_mskinvold
#SBATCH --output=fixMMCLcv1_r18c10v1_opt10.lp_2a80mc_mskinvoldkdagf_1-%j-%a.out # set the output file name for each task. The %j is replaced by the job ID and %a is replaced by the task ID.

#sgf not working? later try v1?
#msk we test with mask_ratios 8 selections
mask_ratios=(0.95 0.85 0.75 0.50 0.35 0.25 0.15 0.05) #8
# ema_ratios=(0.9 0.8 0.5 0.2 0.1 0.05 0.01) #7
#20. 15. 10. 5. 3. 2. 1. .8 .5 .1||  kd!
# 15. 10. 3. 2. 1. .8 .5 .1||     l2sfmx!
# 20. 15. 10. 3. 1. .8 .5||     cssfmx!
echo "maskratios:" $mask_ratios[$SLURM_ARRAY_TASK_ID]
# echo "ema_ratios:" ${ema_ratios[$SLURM_ARRAY_TASK_ID]}
module purge
cd /scratch/yw6594/cf/mmcl
source /vast/yw6594/miniforge3/bin/activate cl
# jobname: MMCL_KDrank_emamsk_res18_cifar10 
# test_res_msk_var.sh  ${mask_ratios[$SLURM_ARRAY_TASK_ID]} v1
# bash /scratch/yw6594/cf/mmcl/tools/cl_trails/wandb_test/runwd_res_msk_var.sh ${mask_ratios[$SLURM_ARRAY_TASK_ID]} 'v1' 's' #MMCL_kdranks_mskinvold MMCL_r18c10v1_clsomskinvoldkds_25
# bash /scratch/yw6594/cf/mmcl/tools/cl_trails/wandb_test/runwd_res_msk_varl2.sh ${mask_ratios[$SLURM_ARRAY_TASK_ID]} 'v2' 'agf' #MMCL_L2ranks_mskinvold MMCL_r18c10v1_clsomskinvoldl2s_25
# bash /scratch/yw6594/cf/mmcl/tools/cl_trails/wandb_test/runwd_res_mskall_var.sh ${mask_ratios[$SLURM_ARRAY_TASK_ID]} 'v1' 's' #MMCL_kdranks_mskinvold MMCL_r18c10v1_mskinvoldkds_25
# bash /scratch/yw6594/cf/mmcl/tools/cl_trails/wandb_test/runwd_res_mskall_varcv1.sh ${mask_ratios[$SLURM_ARRAY_TASK_ID]} 'v1' 's'
# bash /scratch/yw6594/cf/mmcl/tools/cl_trails/wandb_test/runwd_res_mskall_varcv1lp.sh ${mask_ratios[$SLURM_ARRAY_TASK_ID]} 'v2' 'agf'
bash /scratch/yw6594/cf/mmcl/tools/cl_trails/wandb_test/mskalllp/runwd_res_mskall_varcv1lpratio.sh ${mask_ratios[$SLURM_ARRAY_TASK_ID]} 'v1' 'agf'
# bash /scratch/yw6594/cf/mmcl/tools/cl_trails/wandb_test/mskalllp/runwd_res_mskclso_varcv1lpratio.sh ${mask_ratios[$SLURM_ARRAY_TASK_ID]} 'v1' 'agf'

bash /scratch/yw6594/cf/mmcl/tools/cl_trails/clean_artifacts.sh
#v1: cssgf csa  a |
#v2: csa csagf |cssgf ?
# ----
#v3: msk only inv use previous classifier as target? +?
#v5: msk only inv use previous classifier as target -?
# /scratch/yw6594/out/cl_test/mskinvold/mskall/kdbasefix/1_l2
#
# del s 0acc to acc?
# try kd
# try longer epochs for msk
#