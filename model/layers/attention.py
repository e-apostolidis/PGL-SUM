# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np


class SelfAttention(nn.Module):
    def __init__(self, input_size=1024, output_size=1024, freq=10000, heads=1, pos_enc=None):
        """ The basic (multi-head) Attention 'cell' containing the learnable parameters of Q, K and V

        :param int input_size: Feature input size of Q, K, V.
        :param int output_size: Feature -hidden- size of Q, K, V.
        :param int freq: The frequency of the sinusoidal positional encoding.
        :param int heads: Number of heads for the attention module.
        :param str | None pos_enc: The type of the positional encoding [supported: Absolute, Relative].
        """
        super(SelfAttention, self).__init__()

        self.permitted_encodings = ["absolute", "relative"]
        if pos_enc is not None:
            pos_enc = pos_enc.lower()
            assert pos_enc in self.permitted_encodings, f"Supported encodings: {*self.permitted_encodings,}"

        self.input_size = input_size
        self.output_size = output_size
        self.heads = heads
        self.pos_enc = pos_enc
        self.freq = freq
        self.Wk, self.Wq, self.Wv = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for _ in range(self.heads):
            self.Wk.append(nn.Linear(in_features=input_size, out_features=output_size//heads, bias=False))
            self.Wq.append(nn.Linear(in_features=input_size, out_features=output_size//heads, bias=False))
            self.Wv.append(nn.Linear(in_features=input_size, out_features=output_size//heads, bias=False))
        self.out = nn.Linear(in_features=output_size, out_features=input_size, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.drop = nn.Dropout(p=0.5)

    def getAbsolutePosition(self, T):
        """Calculate the sinusoidal positional encoding based on the absolute position of each considered frame.
        Based on 'Attention is all you need' paper (https://arxiv.org/abs/1706.03762)

        :param int T: Number of frames contained in Q, K and V
        :return: Tensor with shape [T, T]
        """
        freq = self.freq
        d = self.input_size

        pos = torch.tensor([k for k in range(T)], device=self.out.weight.device)
        i = torch.tensor([k for k in range(T//2)], device=self.out.weight.device)

        # Reshape tensors each pos_k for each i indices
        pos = pos.reshape(pos.shape[0], 1)
        pos = pos.repeat_interleave(i.shape[0], dim=1)
        i = i.repeat(pos.shape[0], 1)

        AP = torch.zeros(T, T, device=self.out.weight.device)
        AP[pos, 2*i] = torch.sin(pos / freq ** ((2 * i) / d))
        AP[pos, 2*i+1] = torch.cos(pos / freq ** ((2 * i) / d))
        return AP

    def getRelativePosition(self, T):
        """Calculate the sinusoidal positional encoding based on the relative position of each considered frame.
        r_pos calculations as here: https://theaisummer.com/positional-embeddings/

        :param int T: Number of frames contained in Q, K and V
        :return: Tensor with shape [T, T]
        """
        freq = self.freq
        d = 2 * T
        min_rpos = -(T - 1)

        i = torch.tensor([k for k in range(T)], device=self.out.weight.device)
        j = torch.tensor([k for k in range(T)], device=self.out.weight.device)

        # Reshape tensors each i for each j indices
        i = i.reshape(i.shape[0], 1)
        i = i.repeat_interleave(i.shape[0], dim=1)
        j = j.repeat(i.shape[0], 1)

        # Calculate the relative positions
        r_pos = j - i - min_rpos

        RP = torch.zeros(T, T, device=self.out.weight.device)
        idx = torch.tensor([k for k in range(T//2)], device=self.out.weight.device)
        RP[:, 2*idx] = torch.sin(r_pos[:, 2*idx] / freq ** ((i[:, 2*idx] + j[:, 2*idx]) / d))
        RP[:, 2*idx+1] = torch.cos(r_pos[:, 2*idx+1] / freq ** ((i[:, 2*idx+1] + j[:, 2*idx+1]) / d))
        return RP

    def forward(self, x):
        """ Compute the weighted frame features, based on either the global or local (multi-head) attention mechanism.

        :param torch.tensor x: Frame features with shape [T, input_size]
        :return: A tuple of:
                    y: Weighted features based on the attention weights, with shape [T, input_size]
                    att_weights : The attention weights (before dropout), with shape [T, T]
        """
        outputs = []
        for head in range(self.heads):
            K = self.Wk[head](x)
            Q = self.Wq[head](x)
            V = self.Wv[head](x)

            # Q *= 0.06                       # scale factor VASNet
            # Q /= np.sqrt(self.output_size)  # scale factor (i.e 1 / sqrt(d_k) )
            energies = torch.matmul(Q, K.transpose(1, 0))
            if self.pos_enc is not None:
                if self.pos_enc == "absolute":
                    AP = self.getAbsolutePosition(T=energies.shape[0])
                    energies = energies + AP
                elif self.pos_enc == "relative":
                    RP = self.getRelativePosition(T=energies.shape[0])
                    energies = energies + RP

            att_weights = self.softmax(energies)
            _att_weights = self.drop(att_weights)
            y = torch.matmul(_att_weights, V)

            # Save the current head output
            outputs.append(y)
        y = self.out(torch.cat(outputs, dim=1))
        return y, att_weights.clone()  # for now we don't deal with the weights (probably max or avg pooling)


if __name__ == '__main__':
    pass
    """Uncomment for a quick proof of concept
    model = SelfAttention(input_size=256, output_size=256, pos_enc="absolute").cuda()
    _input = torch.randn(500, 256).cuda()  # [seq_len, hidden_size]
    output, weights = model(_input)
    print(f"Output shape: {output.shape}\tattention shape: {weights.shape}")
    """
