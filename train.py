import os
import sys
import json
import random
import time
import copy
import glob

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from data_utils import Dataset
from models.transformer import TransformerForecasting
from models.lstm import LSTMForecasting


def train(
    data_csv_path: str,
    feature_target_names_path: str,
    output_json_path: str,
    model_name: str,
    log_dir: str,
    model_dir: str,
    auxiliary_feat: list,
    use_periodic_encoder: bool,
    use_periodic_as_feat: bool,
    seq_len: int = 120,
    horizon: int = 30,
    batch_size: int = 64,
    epochs: int = 200,
    lr: float = 0.001,
    gpu: int = 3,
):
    
    device = 1 if torch.cuda.is_available() else None
    
    data = pd.read_csv(data_csv_path)

    with open(feature_target_names_path) as f:
        feature_target_names = json.load(f)

    data_train = data[~data[feature_target_names["target"]].isna()]

    grp_by_train = data_train.groupby(by=feature_target_names["group_by_key"])

    groups = list(grp_by_train.groups)

    full_groups = [
        grp for grp in groups if grp_by_train.get_group(grp).shape[0] > 2 * horizon
    ]

    train_data = Dataset(
        groups=full_groups,
        grp_by=grp_by_train,
        split="train",
        features=copy.copy(auxiliary_feat),
        target=feature_target_names["target"],
        seq_len=seq_len,
        horizon=horizon,
        use_periodic_as_feat=use_periodic_as_feat
    )

    val_data = Dataset(
        groups=full_groups,
        grp_by=grp_by_train,
        split="val",
        features=copy.copy(auxiliary_feat),
        target=feature_target_names["target"],
        seq_len=seq_len,
        horizon=horizon,
        use_periodic_as_feat=use_periodic_as_feat
    )

    in_channels = len(auxiliary_feat) + use_periodic_as_feat + 1
    assert in_channels == train_data[0][0].size(1) - 1 == val_data[0][0].size(1) - 1
    print(f"len(in_channels) - {in_channels}")

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=10,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        num_workers=10,
        shuffle=False,
    )

    if model_name == 'transformer':
        model = TransformerForecasting(
            n_encoder_inputs=in_channels,
            n_decoder_inputs=in_channels,
            h_channels=512,
            out_channels=1,
            lr=lr,
            dropout=0.1,
            use_periodic_encoder=use_periodic_encoder,
        )
        
    elif model_name == 'lstm':
        model = LSTMForecasting(
            n_encoder_inputs=in_channels,
            n_decoder_inputs=in_channels,
            h_channels=512,
            out_channels=1,
            lr=lr,
            dropout=0.1,)
    else:
        raise NotImplementedError
        
    tensorboard_logger = TensorBoardLogger(
        save_dir=log_dir,
        name='tensorboard_log',
    )

    csv_logger = CSVLogger(
        save_dir=log_dir, 
        name='csv_log', 
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss",
        mode="min",
        dirpath=model_dir,
        filename='{epoch}-{val_loss:.5f}',
    )
    
    earlystop_callback = EarlyStopping(
        monitor='valid_loss', 
        min_delta=0.00, 
        mode='min',
        patience=20, 
        verbose=True, 
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=[gpu],
        progress_bar_refresh_rate=0.5,
        logger=[tensorboard_logger, csv_logger],
        callbacks=[checkpoint_callback, earlystop_callback],
    )
    
    trainer.fit(model, train_loader, val_loader)

    result_val = trainer.test(test_dataloaders=val_loader)

    output_json = {
        "model_name": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "horizon": horizon,
        "val_loss": result_val[0]["test_loss"],
        "best_model_path": checkpoint_callback.best_model_path,
    }

    if output_json_path is not None:
        with open(output_json_path, "w") as f:
            json.dump(output_json, f, indent=4)

    return output_json


if __name__ == "__main__":
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_name", type=str, required=True)
    
    parser.add_argument("--data_csv_path", default="data/")
    parser.add_argument("--feature_target_names_path", default="data/")
    parser.add_argument("--result_dir", default="results/")
    
    parser.add_argument("--auxiliary_feat", type=str, default="", help="Default: not using auxiliary features")
    parser.add_argument("--use_periodic_encoder", action="store_true", help="Default: False")
    parser.add_argument("--use_periodic_as_feat", action="store_true", help="Default: False")
        
    parser.add_argument("--seq_len", type=int, default=120)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gpu", type=int, default=3)
    
    args = parser.parse_args()
    for arg in vars(args):
        print(f'{arg} - {getattr(args, arg)}')

    auxiliary_feat = ["day_of_week", "day_of_month", "day_of_year", "month", "week_of_year", "year"]
    auxiliary_feat = [auxiliary_feat[int(i)] for i in list(args.auxiliary_feat)]
    print(f'auxiliary_feat - {auxiliary_feat}')
        
    if args.model_name == "lstm":   
        assert args.use_periodic_encoder == False , "cannot use periodic encoder in LSTM"
    
    """ checking if the setting has been trained """
    fname = "{}_aux{}_penc{}_pfeat{}_{}_*".format(
        args.model_name, 
        args.auxiliary_feat,
        1 if args.use_periodic_encoder else 0,
        1 if args.use_periodic_as_feat else 0,    
        args.data_name)
    
    fname = glob.glob(f"{args.result_dir}/{fname}")
    if len(fname) > 0:
        print(f'This setting has been trained for {len(fname)} time(s).')
        sys.exit(0)
    
    model_name = "{}_aux{}_penc{}_pfeat{}_{}_{}".format(
        args.model_name, 
        args.auxiliary_feat,
        1 if args.use_periodic_encoder else 0,
        1 if args.use_periodic_as_feat else 0,    
        args.data_name,
        int(time.time()))
    print(f'model_name - {model_name}')
        
    data_csv_path = os.path.join(args.data_csv_path, f"preprocess_data_{args.data_name}.csv")
    feature_target_names_path = os.path.join(args.feature_target_names_path, f"config_{args.data_name}.json")
    result_dir = os.path.join(args.result_dir, model_name)    
    log_dir = os.path.join(result_dir, "logs")
    model_dir = os.path.join(result_dir, "models")
    output_json_path = os.path.join(result_dir, "trained_config.json")
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    else:
        print(f'{result_dir} has already exists! Please change the model name.')
        sys.exit(0)
    
    print(f'data_csv_path - {data_csv_path}')
    print(f'feature_target_names_path - {feature_target_names_path}')    
    print(f'result_dir - {result_dir}')
    print(f'log_dir - {log_dir}')
    print(f'model_dir - {model_dir}')
    print(f'output_json_path - {output_json_path}')   
            
    train(
        data_csv_path=data_csv_path,
        feature_target_names_path=feature_target_names_path,
        output_json_path=output_json_path,
        model_name=args.model_name,
        log_dir=log_dir,
        model_dir=model_dir,
        auxiliary_feat=auxiliary_feat,
        use_periodic_encoder=args.use_periodic_encoder,
        use_periodic_as_feat=args.use_periodic_as_feat,
        seq_len=args.seq_len,
        horizon=args.horizon,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        gpu=args.gpu,
    )