import torch
from torch import nn as nn


class PatchClassifier(nn.Module):
    def __init__(self):
        super(PatchClassifier, self).__init__()
        self.HIDDEN_DIM = 768
        self.before_linear = nn.Linear(5 * self.HIDDEN_DIM, self.HIDDEN_DIM)
        self.after_linear = nn.Linear(5 * self.HIDDEN_DIM, self.HIDDEN_DIM)
        self.combine = nn.Linear(2 * self.HIDDEN_DIM, self.HIDDEN_DIM)
        self.out_proj = nn.Linear(self.HIDDEN_DIM, 2)

    def forward(self, before_batch, after_batch):
        d1, d2, d3 = before_batch.shape
        before_batch = torch.reshape(before_batch, (d1, d2*d3))
        after_batch = torch.reshape(after_batch, (d1, d2*d3))

        before = self.before_linear(before_batch)
        after = self.after_linear(after_batch)
        combined = self.linear(torch.cat([before, after], axis=1))
        out = self.out_proj(combined)

        return out
