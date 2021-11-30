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

# python test.py --model_name lstm_aux0_penc0_pfeat0_1637795369 --num_test 2000
# python test.py --model_name lstm_aux0_penc0_pfeat1_1637892971 --num_test 2000
# python test.py --model_name lstm_aux1_penc0_pfeat0_1637792135 --num_test 2000
# python test.py --model_name lstm_aux1_penc0_pfeat1_1637827717 --num_test 2000

python test.py --model_name lstm_aux1_penc0_pfeat1_p10_5000_1638189182 --num_test 1000
python test.py --model_name lstm_aux1_penc0_pfeat1_p7_5000_1638164884 --num_test 1000
python test.py --model_name lstm_aux0_penc0_pfeat0_p7_5000_1638152573 --num_test 1000
python test.py --model_name lstm_aux1_penc0_pfeat0_p7_5000_1638152573 --num_test 1000
python test.py --model_name lstm_aux0_penc0_pfeat1_p7_5000_1638167639 --num_test 1000
python test.py --model_name lstm_aux0_penc0_pfeat0_p10_5000_1638172477 --num_test 1000
python test.py --model_name lstm_aux1_penc0_pfeat0_p10_5000_1638182825 --num_test 1000
