# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random
import json
import h5py
from tqdm import tqdm, trange
from layers.summarizer import PGL_SUM
from utils import TensorboardWriter


class Solver(object):
    def __init__(self, config=None, train_loader=None, test_loader=None):
        """Class that Builds, Trains and Evaluates PGL-SUM model"""
        # Initialize variables to None, to be safe
        self.model, self.optimizer, self.writer = None, None, None

        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader

        # Set the seed for generating reproducible random numbers
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
            np.random.seed(self.config.seed)
            random.seed(self.config.seed)

    def build(self):
        """ Function for constructing the PGL-SUM model of its key modules and parameters."""
        # Model creation
        self.model = PGL_SUM(input_size=self.config.input_size,
                             output_size=self.config.input_size,
                             num_segments=self.config.n_segments,
                             heads=self.config.heads,
                             fusion=self.config.fusion,
                             pos_enc=self.config.pos_enc).to(self.config.device)
        if self.config.init_type is not None:
            self.init_weights(self.model, init_type=self.config.init_type, init_gain=self.config.init_gain)

        if self.config.mode == 'train':
            # Optimizer initialization
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.l2_req)
            self.writer = TensorboardWriter(str(self.config.log_dir))

    @staticmethod
    def init_weights(net, init_type="xavier", init_gain=1.4142):
        """ Initialize 'net' network weights, based on the chosen 'init_type' and 'init_gain'.

        :param nn.Module net: Network to be initialized.
        :param str init_type: Name of initialization method: normal | xavier | kaiming | orthogonal.
        :param float init_gain: Scaling factor for normal.
        """
        for name, param in net.named_parameters():
            if 'weight' in name and "norm" not in name:
                if init_type == "normal":
                    nn.init.normal_(param, mean=0.0, std=init_gain)
                elif init_type == "xavier":
                    nn.init.xavier_uniform_(param, gain=np.sqrt(2.0))  # ReLU activation function
                elif init_type == "kaiming":
                    nn.init.kaiming_uniform_(param, mode="fan_in", nonlinearity="relu")
                elif init_type == "orthogonal":
                    nn.init.orthogonal_(param, gain=np.sqrt(2.0))      # ReLU activation function
                else:
                    raise NotImplementedError(f"initialization method {init_type} is not implemented.")
            elif 'bias' in name:
                nn.init.constant_(param, 0.1)

    criterion = nn.MSELoss()

    def train(self):
        """ Main function to train the PGL-SUM model. """
        for epoch_i in trange(self.config.n_epochs, desc='Epoch', ncols=80):
            self.model.train()

            loss_history = []
            num_batches = int(len(self.train_loader) / self.config.batch_size)  # full-batch or mini batch
            iterator = iter(self.train_loader)
            for _ in trange(num_batches, desc='Batch', ncols=80, leave=False):
                # ---- Training ... ----#
                if self.config.verbose:
                    tqdm.write('Time to train the model...')

                self.optimizer.zero_grad()
                for _ in trange(self.config.batch_size, desc='Video', ncols=80, leave=False):
                    frame_features, target = next(iterator)

                    frame_features = frame_features.to(self.config.device)
                    target = target.to(self.config.device)

                    output, weights = self.model(frame_features.squeeze(0))
                    loss = self.criterion(output.squeeze(0), target.squeeze(0))

                    if self.config.verbose:
                        tqdm.write(f'[{epoch_i}] loss: {loss.item()}')

                    loss.backward()
                    loss_history.append(loss.data)
                # Update model parameters every 'batch_size' iterations
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
                self.optimizer.step()

            # Mean loss of each training step
            loss = torch.stack(loss_history).mean()

            # Plot
            if self.config.verbose:
                tqdm.write('Plotting...')

            self.writer.update_loss(loss, epoch_i, 'loss_epoch')
            # Uncomment to save parameters at checkpoint
            if not os.path.exists(self.config.save_dir):
                os.makedirs(self.config.save_dir)
            # ckpt_path = str(self.config.save_dir) + f'/epoch-{epoch_i}.pkl'
            # tqdm.write(f'Save parameters at {ckpt_path}')
            # torch.save(self.model.state_dict(), ckpt_path)

            self.evaluate(epoch_i)

    def evaluate(self, epoch_i, save_weights=False):
        """ Saves the frame's importance scores for the test videos in json format.

        :param int epoch_i: The current training epoch.
        :param bool save_weights: Optionally, the user can choose to save the attention weights in a (large) h5 file.
        """
        self.model.eval()

        weights_save_path = self.config.score_dir.joinpath("weights.h5")
        out_scores_dict = {}
        for frame_features, video_name in tqdm(self.test_loader, desc='Evaluate', ncols=80, leave=False):
            # [seq_len, input_size]
            frame_features = frame_features.view(-1, self.config.input_size).to(self.config.device)

            with torch.no_grad():
                scores, attn_weights = self.model(frame_features)  # [1, seq_len]
                scores = scores.squeeze(0).cpu().numpy().tolist()
                attn_weights = attn_weights.cpu().numpy()

                out_scores_dict[video_name] = scores

            if not os.path.exists(self.config.score_dir):
                os.makedirs(self.config.score_dir)

            scores_save_path = self.config.score_dir.joinpath(f"{self.config.video_type}_{epoch_i}.json")
            with open(scores_save_path, 'w') as f:
                if self.config.verbose:
                    tqdm.write(f'Saving score at {str(scores_save_path)}.')
                json.dump(out_scores_dict, f)
            scores_save_path.chmod(0o777)

            if save_weights:
                with h5py.File(weights_save_path, 'a') as weights:
                    weights.create_dataset(f"{video_name}/epoch_{epoch_i}", data=attn_weights)


if __name__ == '__main__':
    pass
