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

# python test.py --model_name transformer_aux0_penc0_pfeat0_1637787108 --num_test 2000
# python test.py --model_name transformer_aux0_penc0_pfeat1_1637812993 --num_test 2000
# python test.py --model_name transformer_aux0_penc1_pfeat0_1637781703 --num_test 2000
# python test.py --model_name transformer_aux0_penc1_pfeat1_1637781703 --num_test 2000
# python test.py --model_name transformer_aux1_penc0_pfeat0_1637868980 --num_test 2000
# python test.py --model_name transformer_aux1_penc0_pfeat1_1637838954 --num_test 2000
# python test.py --model_name transformer_aux1_penc1_pfeat0_1637805164 --num_test 2000
# python test.py --model_name transformer_aux1_penc1_pfeat1_1637818343 --num_test 2000

python test.py --model_name transformer_aux1_penc0_pfeat1_p10_5000_1638183260 --num_test 1000
python test.py --model_name transformer_aux0_penc0_pfeat0_p10_5000_1638188294 --num_test 1000
python test.py --model_name transformer_aux1_penc1_pfeat1_p7_5000_1638190548 --num_test 1000
python test.py --model_name transformer_aux0_penc1_pfeat0_p7_5000_1638152733 --num_test 1000
python test.py --model_name transformer_aux0_penc0_pfeat1_p10_5000_1638152778 --num_test 1000
python test.py --model_name transformer_aux0_penc1_pfeat1_p7_5000_1638152923 --num_test 1000
python test.py --model_name transformer_aux1_penc0_pfeat0_p10_5000_1638162417 --num_test 1000
python test.py --model_name transformer_aux1_penc1_pfeat0_p7_5000_1638163121 --num_test 1000
python test.py --model_name transformer_aux0_penc0_pfeat1_p7_5000_1638164780 --num_test 1000
python test.py --model_name transformer_aux1_penc0_pfeat1_p7_5000_1638169717 --num_test 1000
python test.py --model_name transformer_aux1_penc0_pfeat0_p7_5000_1638177176 --num_test 1000
python test.py --model_name transformer_aux0_penc0_pfeat0_p7_5000_1638181986 --num_test 1000
