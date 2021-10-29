# -*- coding: utf-8 -*-
import torch
import numpy as np
from generate_summary import generate_summary
from evaluation_metrics import evaluate_summary
from layers.summarizer import PGL_SUM
from os import listdir
from os.path import isfile, join
import h5py
import json
import argparse


def inference(model, data_path, keys, eval_method):
    """ Used to inference a pretrained `model` on the `keys` test videos, based on the `eval_method` criterion; using
        the dataset located in `data_path'.

        :param nn.Module model: Pretrained model to be inferenced.
        :param str data_path: File path for the dataset in use.
        :param list keys: Containing the test video keys of the used data split.
        :param str eval_method: The evaluation method in use {SumMe: max, TVSum: avg}
    """
    model.eval()
    video_fscores = []
    for video in keys:
        with h5py.File(data_path, "r") as hdf:
            # Input features for inference
            frame_features = torch.Tensor(np.array(hdf[f"{video}/features"])).view(-1, 1024)
            # Input need for evaluation
            user_summary = np.array(hdf[f"{video}/user_summary"])
            sb = np.array(hdf[f"{video}/change_points"])
            n_frames = np.array(hdf[f"{video}/n_frames"])
            positions = np.array(hdf[f"{video}/picks"])

        with torch.no_grad():
            scores, _ = model(frame_features)  # [1, seq_len]
            scores = scores.squeeze(0).cpu().numpy().tolist()
            summary = generate_summary([sb], [scores], [n_frames], [positions])[0]
            f_score = evaluate_summary(summary, user_summary, eval_method)
            video_fscores.append(f_score)
    print(f"Trained for split: {split_id} achieved an F-score of {np.mean(video_fscores):.2f}%")


if __name__ == "__main__":
    # arguments to run the script
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='SumMe', help="Dataset to be used. Supported: {SumMe, TVSum}")
    parser.add_argument("--table", type=str, default='4', help="Table to be reproduced. Supported: {3, 4}")

    args = vars(parser.parse_args())
    dataset = args["dataset"]
    table = args["table"]

    eval_metric = 'avg' if dataset.lower() == 'tvsum' else 'max'
    for split_id in range(5):
        # Model data
        model_path = f"../PGL-SUM/inference/pretrained_models/table{table}_models/{dataset}/split{split_id}"
        model_file = [f for f in listdir(model_path) if isfile(join(model_path, f))]

        # Read current split
        split_file = f"../PGL-SUM/data/splits/{dataset.lower()}_splits.json"
        with open(split_file) as f:
            data = json.loads(f.read())
            test_keys = data[split_id]["test_keys"]

        # Dataset path
        dataset_path = f"../PGL-SUM/data/{dataset}/eccv16_dataset_{dataset.lower()}_google_pool5.h5"

        # Create model with paper reported configuration
        trained_model = PGL_SUM(input_size=1024, output_size=1024, num_segments=4, heads=8,
                                fusion="add", pos_enc="absolute")
        trained_model.load_state_dict(torch.load(join(model_path, model_file[-1])))
        inference(trained_model, dataset_path, test_keys, eval_metric)
