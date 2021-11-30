#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lin00786@umn.edu
#SBATCH -p amdlarge

module load python3

source activate torch-env

cd /home/yaoyi/lin00786/transformer-adding_periodic/

python scripts/generate_time_series.py --data_path data/data_p11_20000.csv --num_samples 20000 --periods 11
python scripts/preprocess_time_series.py --data_path data/data_p11_20000.csv --preprocess_data_path data/preprocess_data_p11_20000.csv --config_path data/config_p11_20000.json

python scripts/generate_time_series.py --data_path data/data_p112337_20000.csv --num_samples 20000 --periods 11 23 37
python scripts/preprocess_time_series.py --data_path data/data_p112337_20000.csv --preprocess_data_path data/preprocess_data_p112337_20000.csv --config_path data/config_p112337_20000.json