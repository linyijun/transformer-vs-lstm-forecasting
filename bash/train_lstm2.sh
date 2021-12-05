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

python train.py --model_name lstm --data_name p71428_20000
python train.py --model_name lstm --data_name p71428_20000 --auxiliary_feat 35
python train.py --model_name lstm --data_name p71428_20000 --auxiliary_feat 012345
