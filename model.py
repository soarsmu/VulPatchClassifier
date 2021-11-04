from transformers import RobertaTokenizer, RobertaModel
import torch
from torch import nn as nn


class PatchClassifier(nn.Module):
    def __init__(self):
        super(PatchClassifier, self).__init__()
        self.HIDDEN_DIM = 768
        self.linear = nn.Linear(2 * self.HIDDEN_DIM, self.HIDDEN_DIM)
        self.out_proj = nn.Linear(self.HIDDEN_DIM, 2)

    def forward(self, before_batch, after_batch):
        combined = self.linear(torch.cat([before_batch, after_batch], axis=1))
        out = self.out_proj(combined)

        return out
