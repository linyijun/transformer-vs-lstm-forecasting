#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lin00786@umn.edu
#SBATCH -p v100

module load python3

source activate torch-env

cd /home/yaoyi/lin00786/transformer-vs-lstm-forecasting/

# auxiliary_feat = ["day_of_week", "day_of_month", "day_of_year", "month", "week_of_year", "year"]

python train.py --model_name transformer --data_name p7_10000 --use_periodic_as_feat --use_periodic_encoder
python train.py --model_name transformer --data_name p7_10000 --auxiliary_feat 35 --use_periodic_as_feat --use_periodic_encoder
python train.py --model_name transformer --data_name p7_10000 --auxiliary_feat 012345 --use_periodic_as_feat --use_periodic_encoder
