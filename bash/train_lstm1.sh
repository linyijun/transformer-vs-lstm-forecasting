#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lin00786@umn.edu
#SBATCH -p v100

module load python3

source activate torch-env

cd /home/yaoyi/lin00786/transformer-adding-periodic/

python train.py --model_name lstm --data_name p7_5000 --batch_size 64 
python train.py --model_name lstm --data_name p7_5000 --batch_size 64 --use_auxiliary --use_periodic_as_feat
python train.py --model_name lstm --data_name p10_5000 --batch_size 64
python train.py --model_name lstm --data_name p10_5000 --batch_size 64 --use_auxiliary --use_periodic_as_feat
