# -*- coding: utf-8 -*-
import torch
import numpy as np
import csv
import json
import sys

# with args (example usage: python choose_best_epoch.py <path_to_experiment> TVSum)
exp_path = sys.argv[1]
dataset = sys.argv[2]
''' without args
exp_path = "../PGL-SUM/Summaries/PGL-SUM/exp1"
dataset = "SumMe"
'''


def train_logs(log_file):
    """
    Choose and return the epoch based on the training loss. The criterion is based on the smoothness of the loss changes
    (1st derivative). Calculate the -percentage wise- difference to get to the global min. Similarly for the first local
    min. Finally, choose the smaller one (by absolute value).
    The intuition behind that criterion is to check if the dataset can handle steep changes.
    :param str log_file: Path to the saved csv file containing the loss information.
    :return: The epoch of the best model.
    """
    losses = {}
    losses_names = []

    # Read the csv file with the training losses
    with open(log_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for (i, row) in enumerate(csv_reader):
            if i == 0:
                for col in range(len(row)):
                    losses[row[col]] = []
                    losses_names.append(row[col])
            else:
                for col in range(len(row)):
                    losses[losses_names[col]].append(float(row[col]))

    # criterion: MSE loss of gtscores
    loss = losses["loss_epoch"]

    START_EPOCH, tol = 15, 1
    cand_epoch, cand_val = 0, 0
    for i in range(START_EPOCH, len(loss)-1):
        diff = (loss[i+1]-loss[i])/loss[i+1] * 100
        if diff <= -tol:
            cand_epoch = i+1  # take the next epoch from that good (negative) change in the loss curve
            cand_val = diff
            break
        if diff >= tol:
            cand_epoch = i    # take the previous epoch from that bad (positive) change in the loss curve
            cand_val = diff
            break

    # Find the absolute minimum
    criterion = torch.tensor(loss)
    argmin_epoch = torch.argmin(criterion).item()
    argmin_diff = (loss[argmin_epoch]-loss[argmin_epoch-1])/loss[argmin_epoch] * 100

    if abs(argmin_diff) < abs(cand_val):
        epoch = argmin_epoch
    else:
        epoch = cand_epoch
    return epoch+1


all_fscores = np.zeros(5, dtype=float)
for split in range(0, 5):
    results_file = exp_path + "/" + dataset + "/results/split" + str(split) + "/f_scores.txt"
    log = exp_path + "/" + dataset + "/logs/split" + str(split) + "/scalars.csv"

    # read F-Scores
    with open(results_file) as f:
        f_scores = f.read().strip()
        if "\n" in f_scores:
            f_scores = f_scores.splitlines()
        else:
            f_scores = json.loads(f_scores)
        f_scores = [float(f_score) for f_score in f_scores]
        selected_epoch = train_logs(log)
        all_fscores[split] = np.round(f_scores[selected_epoch], 2)
        print(f"Split: {split} -> Criterion Fscore: {all_fscores[split]} @ epoch: {selected_epoch}")

avg_fscore = np.mean(all_fscores)
print(f"Average Fscore: {avg_fscore}")
