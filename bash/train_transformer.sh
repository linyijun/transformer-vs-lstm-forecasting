#!/bin/bash -l

cd /data/yijun/transformer-vs-lstm-forecasting/

# auxiliary_feat = ["day_of_week", "day_of_month", "day_of_year", "month", "week_of_year", "year"]

# python train.py --model_name transformer --data_name p7_10000 --gpu 2
# python train.py --model_name transformer --data_name p11_10000 --gpu 2
# python train.py --model_name transformer --data_name p71428_10000 --gpu 2
# python train.py --model_name transformer --data_name p112337_10000 --gpu 2

python train.py --model_name transformer --data_name p7_10000 --use_periodic_encoder --gpu 2
python train.py --model_name transformer --data_name p11_10000 --use_periodic_encoder --gpu 2
python train.py --model_name transformer --data_name p71428_10000 --use_periodic_encoder --gpu 2
python train.py --model_name transformer --data_name p112337_10000 --use_periodic_encoder --gpu 2

python train.py --model_name transformer --data_name p7_10000 --use_periodic_as_feat --gpu 2
python train.py --model_name transformer --data_name p11_10000 --use_periodic_as_feat --gpu 2
python train.py --model_name transformer --data_name p71428_10000 --use_periodic_as_feat --gpu 2
python train.py --model_name transformer --data_name p112337_10000 --use_periodic_as_feat --gpu 2


python train.py --model_name transformer --data_name p7_10000 --auxiliary_feat 35 --gpu 2
python train.py --model_name transformer --data_name p11_10000 --auxiliary_feat 35 --gpu 2
python train.py --model_name transformer --data_name p71428_10000 --auxiliary_feat 35 --gpu 2
python train.py --model_name transformer --data_name p112337_10000 --auxiliary_feat 35 --gpu 2

python train.py --model_name transformer --data_name p7_10000 --auxiliary_feat 012345 --gpu 2
python train.py --model_name transformer --data_name p11_10000 --auxiliary_feat 012345 --gpu 2
python train.py --model_name transformer --data_name p71428_10000 --auxiliary_feat 012345 --gpu 2
python train.py --model_name transformer --data_name p112337_10000 --auxiliary_feat 012345 --gpu 2


python train.py --model_name transformer --data_name p7_10000 --use_periodic_encoder --auxiliary_feat 35 --gpu 2
python train.py --model_name transformer --data_name p11_10000 --use_periodic_encoder --auxiliary_feat 35 --gpu 2
python train.py --model_name transformer --data_name p71428_10000 --use_periodic_encoder --auxiliary_feat 35 --gpu 2
python train.py --model_name transformer --data_name p112337_10000 --use_periodic_encoder --auxiliary_feat 35 --gpu 2

python train.py --model_name transformer --data_name p7_10000 --use_periodic_encoder --auxiliary_feat 012345 --gpu 2
python train.py --model_name transformer --data_name p11_10000 --use_periodic_encoder --auxiliary_feat 012345 --gpu 2
python train.py --model_name transformer --data_name p71428_10000 --use_periodic_encoder --auxiliary_feat 012345 --gpu 2
python train.py --model_name transformer --data_name p112337_10000 --use_periodic_encoder --auxiliary_feat 012345 --gpu 2


python train.py --model_name transformer --data_name p7_10000 --use_periodic_as_feat --auxiliary_feat 35 --gpu 2
python train.py --model_name transformer --data_name p11_10000 --use_periodic_as_feat --auxiliary_feat 35 --gpu 2
python train.py --model_name transformer --data_name p71428_10000 --use_periodic_as_feat --auxiliary_feat 35 --gpu 2
python train.py --model_name transformer --data_name p112337_10000 --use_periodic_as_feat --auxiliary_feat 35 --gpu 2

python train.py --model_name transformer --data_name p7_10000 --use_periodic_as_feat --auxiliary_feat 012345 --gpu 2
python train.py --model_name transformer --data_name p11_10000 --use_periodic_as_feat --auxiliary_feat 012345 --gpu 2
python train.py --model_name transformer --data_name p71428_10000 --use_periodic_as_feat --auxiliary_feat 012345 --gpu 2
python train.py --model_name transformer --data_name p112337_10000 --use_periodic_as_feat --auxiliary_feat 012345 --gpu 2


python train.py --model_name transformer --data_name p7_10000 --use_periodic_encoder --use_periodic_as_feat --gpu 2
python train.py --model_name transformer --data_name p11_10000 --use_periodic_encoder --use_periodic_as_feat --gpu 2
python train.py --model_name transformer --data_name p71428_10000 --use_periodic_encoder --use_periodic_as_feat --gpu 2
python train.py --model_name transformer --data_name p112337_10000 --use_periodic_encoder --use_periodic_as_feat --gpu 2

python train.py --model_name transformer --data_name p7_10000 --use_periodic_encoder --use_periodic_as_feat --auxiliary_feat 35 --gpu 2
python train.py --model_name transformer --data_name p11_10000 --use_periodic_encoder --use_periodic_as_feat --auxiliary_feat 35 --gpu 2
python train.py --model_name transformer --data_name p71428_10000 --use_periodic_encoder --use_periodic_as_feat --auxiliary_feat 35 --gpu 2
python train.py --model_name transformer --data_name p112337_10000 --use_periodic_encoder --use_periodic_as_feat --auxiliary_feat 35 --gpu 2

python train.py --model_name transformer --data_name p7_10000 --use_periodic_encoder --use_periodic_as_feat --auxiliary_feat 012345 --gpu 2
python train.py --model_name transformer --data_name p11_10000 --use_periodic_encoder --use_periodic_as_feat --auxiliary_feat 012345 --gpu 2
python train.py --model_name transformer --data_name p71428_10000 --use_periodic_encoder --use_periodic_as_feat --auxiliary_feat 012345 --gpu 2
python train.py --model_name transformer --data_name p112337_10000 --use_periodic_encoder --use_periodic_as_feat --auxiliary_feat 012345 --gpu 2
