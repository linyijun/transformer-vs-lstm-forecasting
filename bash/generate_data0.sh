#!/bin/bash -l
cd /data/yijun/transformer-vs-lstm-forecasting/

python scripts/generate_time_series.py --data_path data/data_p7_10000.csv --num_samples 10000 --periods 7
python scripts/preprocess_time_series.py --data_path data/data_p7_10000.csv --preprocess_data_path data/preprocess_data_p7_10000.csv --config_path data/config_p7_10000.json

python scripts/generate_time_series.py --data_path data/data_p71428_10000.csv --num_samples 10000 --periods 7 14 28
python scripts/preprocess_time_series.py --data_path data/data_p71428_10000.csv --preprocess_data_path data/preprocess_data_p71428_10000.csv --config_path data/config_p71428_10000.json

python scripts/generate_time_series.py --data_path data/data_p11_10000.csv --num_samples 10000 --periods 11
python scripts/preprocess_time_series.py --data_path data/data_p11_10000.csv --preprocess_data_path data/preprocess_data_p11_10000.csv --config_path data/config_p11_10000.json

python scripts/generate_time_series.py --data_path data/data_p112337_10000.csv --num_samples 10000 --periods 11 23 37
python scripts/preprocess_time_series.py --data_path data/data_p112337_10000.csv --preprocess_data_path data/preprocess_data_p112337_10000.csv --config_path data/config_p112337_10000.json


