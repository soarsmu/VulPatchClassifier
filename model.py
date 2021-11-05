import torch
from torch import nn as nn
import numpy as np
from torch.nn import functional as F


class PatchClassifier(nn.Module):
    def __init__(self):
        super(PatchClassifier, self).__init__()
        self.HIDDEN_DIM = 768
        self.before_linear = nn.Linear(5 * self.HIDDEN_DIM, self.HIDDEN_DIM)
        self.after_linear = nn.Linear(5 * self.HIDDEN_DIM, self.HIDDEN_DIM)
        self.combine = nn.Linear(2 * self.HIDDEN_DIM, self.HIDDEN_DIM)
        self.l1 = nn.Linear(self.HIDDEN_DIM, self.HIDDEN_DIM)
        self.l2 = nn.Linear(self.HIDDEN_DIM, self.HIDDEN_DIM)
        self.l3 = nn.Linear(self.HIDDEN_DIM, self.HIDDEN_DIM)
        self.out_proj = nn.Linear(self.HIDDEN_DIM, 2)

    def forward(self, before_batch, after_batch):
        d1, d2, d3 = before_batch.shape
        before_batch = torch.reshape(before_batch, (d1, d2*d3))
        after_batch = torch.reshape(after_batch, (d1, d2*d3))

        before = self.before_linear(before_batch)
        after = self.after_linear(after_batch)
        combined = self.combine(torch.cat([before, after], axis=1))
        combined = self.l1(combined)
        combined = self.l2(combined)
        combined = self.l3(combined)
        out = self.out_proj(combined)

        return out


class CnnClassifier(nn.Module):
    """An 1D Convulational Neural Network for Sentence Classification."""

    def __init__(self,
                 embed_dim=768,
                 filter_sizes=[2, 3, 4],
                 num_filters=[100, 100, 100],
                 num_classes=2,
                 dropout=0.5):
        super(CnnClassifier, self).__init__()
        self.embed_dim = embed_dim
        print(filter_sizes)
        # Conv Network
        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=self.embed_dim,
                      out_channels=num_filters[i],
                      kernel_size=filter_sizes[i])
            for i in range(len(filter_sizes))
        ])
        # Fully-connected layer and Dropout
        self.fc = nn.Linear(np.sum(num_filters), num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, code):
        """Perform a forward pass through the network.

        Args:
            input_ids (torch.Tensor): A tensor of token ids with shape
                (batch_size, max_sent_length)

        Returns:
            logits (torch.Tensor): Output logits with shape (batch_size,
                n_classes)
        """

        # Get embeddings from `input_ids`. Output shape: (b, max_len, embed_dim)
        x_embed = code

        # Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
        # Output shape: (b, embed_dim, max_len)
        x_reshaped = x_embed.permute(0, 2, 1)

        # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
        x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]

        # Max pooling. Output shape: (b, num_filters[i], 1)
        x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
                       for x_conv in x_conv_list]

        # Concatenate x_pool_list to feed the fully connected layer.
        # Output shape: (b, sum(num_filters))
        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list],
                         dim=1)

        # Compute logits. Output shape: (b, n_classes)
        out = self.fc(self.dropout(x_fc))
        return F.log_softmax(out, -1)