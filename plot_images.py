import json
import os

import matplotlib.pyplot as plt


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--result_dir", default="results/")
    parser.add_argument("--num_test", type=int, default=10)
    args = parser.parse_args()
    
    output_dir = os.path.join(args.result_dir, args.model_name, "outputs")    
    data_for_visualization_path = os.path.join(output_dir, "visualization.json")

    with open(data_for_visualization_path, "r") as f:
        data = json.load(f)

    if not os.path.exists(os.path.join(output_dir, "images")):
        os.makedirs(os.path.join(output_dir, "images"))

    for i, sample in enumerate(data[:args.num_test]):
        hist_size = len(sample["history"])
        gt_size = len(sample["ground_truth"])
        plt.figure()
        plt.plot(range(hist_size), sample["history"], label="History")
        plt.plot(
            range(hist_size, hist_size + gt_size), sample["ground_truth"], label="Ground Truth"
        )
        plt.plot(
            range(hist_size, hist_size + gt_size), sample["prediction"], label="Prediction"
        )

        plt.xlabel("Time")

        plt.ylabel("Time Series")

        plt.legend()

        plt.savefig(os.path.join(output_dir, "images", f"{i}.png"))
        plt.close()