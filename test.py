import os
import json
from typing import Optional
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

import random
from models.transformer import TransformerForecasting
from models.lstm import LSTMForecasting
from data_utils import split_df, Dataset


def smape(true, pred):
    """
    Symmetric mean absolute percentage error
    :param true:
    :param pred:
    :return:
    """
    true = np.array(true)
    pred = np.array(pred)

    smape_val = (
        100
        / pred.size
        * np.sum(2 * (np.abs(true - pred)) / (np.abs(pred) + np.abs(true) + 1e-8))
    )

    return smape_val


def evaluate_regression(true, pred):
    """
    eval mae + smape
    :param true:
    :param pred:
    :return:
    """

    return {"smape": smape(true, pred), "mae": mean_absolute_error(true, pred)}


def test(
    data_csv_path: str,
    feature_target_names_path: str,
    model_name: str,
    model_path: str,    
    test_json_path: str,
    seq_len: int = 120,
    horizon: int = 30,
    data_for_visualization_path: Optional[str] = None,
    num_test: Optional[int] = 100
):
    
    use_auxiliary = int(model_name[model_name.find('aux') + 3])
    use_periodic_encoder = int(model_name[model_name.find('penc') + 4])
    use_periodic_as_feat = int(model_name[model_name.find('pfeat') + 5])
    print("use_auxiliary - ", use_auxiliary)
    print("use_periodic_encoder - ", use_periodic_encoder)
    print("use_periodic_as_feat - ", use_periodic_as_feat)
    
    data = pd.read_csv(data_csv_path)

    with open(feature_target_names_path) as f:
        feature_target_names = json.load(f)
    target = feature_target_names["target"]

    data_train = data[~data[target].isna()]
    grp_by_train = data_train.groupby(by=feature_target_names["group_by_key"])
    groups = list(grp_by_train.groups)

    full_groups = [
        grp for grp in groups if grp_by_train.get_group(grp).shape[0] > horizon
    ]

    test_data = Dataset(
        groups=full_groups,
        grp_by=grp_by_train,
        split="test",
        features=feature_target_names["features"],
        target=feature_target_names["target"],
        seq_len=seq_len,
        horizon=horizon,
        use_auxiliary=use_auxiliary,
        use_periodic_as_feat=use_periodic_as_feat        
    )

    in_channels = len(feature_target_names["features"]) * use_auxiliary + use_periodic_as_feat + 1
    assert in_channels == test_data[0][0].size(1) - 1
    print(f"len(in_channels) - {in_channels}")
    
    if 'transformer' in model_name:
        model = TransformerForecasting(
        n_encoder_inputs=in_channels,
        n_decoder_inputs=in_channels,
        h_channels=512,
        out_channels=1,
        use_periodic_encoder=use_periodic_encoder,)    
        
    elif 'lstm' in model_name:        
        model = LSTMForecasting(
            n_encoder_inputs=in_channels,
            n_decoder_inputs=in_channels,
            h_channels=512,
            out_channels=1,)
    else:
        raise NotImplementedError

    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device)["state_dict"], strict=False)
    
    model.eval()

    gt = []
    baseline_last_known_values = []
    neural_predictions = []

    data_for_visualization = []

    random.seed(1234)
    test_idx = random.sample([i for i in range(len(full_groups))], num_test)
    
    for i, group in tqdm(enumerate(full_groups)):
        
        if i not in test_idx:
            continue
            
        time_series_data = {"history": [], "ground_truth": [], "prediction": []}

        df = grp_by_train.get_group(group)
        src, trg = split_df(df, split="val")
        
        time_series_data["history"] = src[target].tolist()[-seq_len:]
        time_series_data["ground_truth"] = trg[target].tolist()

        last_known_value = src[target].values[-1]

        trg["last_known_value"] = last_known_value

        gt += trg[target].tolist()
        baseline_last_known_values += trg["last_known_value"].tolist()

        src, trg_in, _ = test_data[i]

        src, trg_in = src.unsqueeze(0), trg_in.unsqueeze(0)  # src/trg_in: [1, seq_len/horizon, channels]

        with torch.no_grad():
            prediction = model((src, trg_in[:, :1, :]))
            for j in range(1, horizon):
                last_prediction = prediction[0, -1]
                trg_in[:, j, -2] = last_prediction
                prediction = model((src, trg_in[:, : (j + 1), :]))  # using the prediction as the input

            trg[target + "_predicted"] = (prediction.squeeze().numpy()).tolist()
            neural_predictions += trg[target + "_predicted"].tolist()
            time_series_data["prediction"] = trg[target + "_predicted"].tolist()

        data_for_visualization.append(time_series_data)

    baseline_eval = evaluate_regression(gt, baseline_last_known_values)
    model_eval = evaluate_regression(gt, neural_predictions)

    eval_dict = {
        "Baseline_MAE": baseline_eval["mae"],
        "Baseline_SMAPE": baseline_eval["smape"],
        "Model_MAE": model_eval["mae"],
        "Model_SMAPE": model_eval["smape"],
    }
    
    for k, v in eval_dict.items():
        print(k, v)

    if test_json_path is not None:
        with open(test_json_path, "w") as f:
            json.dump(eval_dict, f, indent=4)

    if data_for_visualization_path is not None:
        with open(data_for_visualization_path, "w") as f:
            json.dump(data_for_visualization, f, indent=4)

    return eval_dict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--num_test", type=int, required=True)  
    
    parser.add_argument("--data_csv_path", default="data/")
    parser.add_argument("--feature_target_names_path", default="data/")
    parser.add_argument("--result_dir", default="results/")
    
    parser.add_argument("--seq_len", type=int, default=120)
    parser.add_argument("--horizon", type=int, default=30)    

    args = parser.parse_args()
    
    data_name = '_'.join(args.model_name.split('_')[-3:-1])
    data_csv_path = os.path.join(args.data_csv_path, f"preprocess_data_{data_name}.csv")
    feature_target_names_path = os.path.join(args.feature_target_names_path, f"config_{data_name}.json")
    result_dir = os.path.join(args.result_dir, args.model_name)
    log_dir = os.path.join(result_dir, "logs")
    model_dir = os.path.join(result_dir, "models")
    trained_json_path = os.path.join(result_dir, "trained_config.json")
    
    """ get the model path with trained config file """
    if not os.path.exists(trained_json_path):
        model_path = os.path.join(model_dir, args.model_name + '.ckpt')
    else:
        with open(trained_json_path) as f:
            model_json = json.load(f)
        model_path = model_json["best_model_path"]
    
    """ output the evaluation results and visualization config """
    output_dir = os.path.join(args.result_dir, args.model_name, "outputs")    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    test_json_path = os.path.join(output_dir, "test.json")
    data_for_visualization_path = os.path.join(output_dir, "visualization.json")
    
    """ test """
    test(
        data_csv_path=data_csv_path,
        feature_target_names_path=feature_target_names_path,
        model_name=args.model_name,
        model_path=model_path,        
        test_json_path=test_json_path,
        seq_len = args.seq_len,
        horizon = args.horizon,
        data_for_visualization_path=data_for_visualization_path,
        num_test = args.num_test,
    )