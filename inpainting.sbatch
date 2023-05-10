#!/bin/sh
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=4
#SBATCH --mem=4000
#SBATCH --gres=gpu:1
module use /opt/insy/modulefiles
module load cuda/11.2 cudnn/11.2-8.1.1.33
srun python inpainting.py --model ce_skip_depth4 --image kate