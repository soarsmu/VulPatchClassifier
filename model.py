from transformers import RobertaTokenizer, RobertaModel
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

        self.out_proj = nn.Linear(self.HIDDEN_DIM, 2)

    def forward(self, before_batch, after_batch):
        d1, d2, d3 = before_batch.shape
        before_batch = torch.reshape(before_batch, (d1, d2*d3))
        after_batch = torch.reshape(after_batch, (d1, d2*d3))

        before = self.before_linear(before_batch)
        after = self.after_linear(after_batch)
        combined = self.combine(torch.cat([before, after], axis=1))

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
        self.fc = nn.Linear(2 * np.sum(num_filters), num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, before_batch, after_batch):
        """Perform a forward pass through the network.

        Args:
            input_ids (torch.Tensor): A tensor of token ids with shape
                (batch_size, max_sent_length)

        Returns:
            logits (torch.Tensor): Output logits with shape (batch_size,
                n_classes)
        """

        # Get embeddings from `input_ids`. Output shape: (b, max_len, embed_dim)
        x_embed_before = before_batch
        # batch, file, hidden_dim

        # Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
        # Output shape: (b, embed_dim, max_len)
        x_reshaped_before = x_embed_before.permute(0, 2, 1)

        # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
        x_conv_list_before = [F.relu(conv1d(x_reshaped_before)) for conv1d in self.conv1d_list]

        # Max pooling. Output shape: (b, num_filters[i], 1)
        x_pool_list_before = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
                       for x_conv in x_conv_list_before]

        # Concatenate x_pool_list to feed the fully connected layer.
        # Output shape: (b, sum(num_filters))
        x_fc_before = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list_before],
                         dim=1)

        # # Compute logits. Output shape: (b, n_classes)
        # out = self.fc(self.dropout(x_fc_before))


        ############################################


        x_embed_after = after_batch

        # Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
        # Output shape: (b, embed_dim, max_len)
        x_reshaped_after = x_embed_after.permute(0, 2, 1)

        # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
        x_conv_list_after = [F.relu(conv1d(x_reshaped_after)) for conv1d in self.conv1d_list]

        # Max pooling. Output shape: (b, num_filters[i], 1)
        x_pool_list_after = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
                       for x_conv in x_conv_list_after]

        # Concatenate x_pool_list to feed the fully connected layer.
        # Output shape: (b, sum(num_filters))
        x_fc_after = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list_after],
                         dim=1)

        # Compute logits. Output shape: (b, n_classes)

        x_fc = torch.cat([x_fc_before, x_fc_after], axis=1)
        out = self.fc(self.dropout(x_fc))

        return out


class VariantTwoClassifier(nn.Module):
    def __init__(self):
        super(VariantTwoClassifier, self).__init__()
        self.HIDDEN_DIM = 768
        self.HIDDEN_DIM_DROPOUT_PROB = 0.3

        self.linear = nn.Linear(5 * self.HIDDEN_DIM, self.HIDDEN_DIM)
        self.relu = nn.ReLU()
        self.drop_out = nn.Dropout(self.HIDDEN_DIM_DROPOUT_PROB)
        self.out_proj = nn.Linear(self.HIDDEN_DIM, 2)

    def forward(self, file_batch):
        d1, d2, d3 = file_batch.shape
        file_batch = torch.reshape(file_batch, (d1, d2*d3))

        commit_embedding = self.linear(file_batch)

        x = commit_embedding
        x = self.relu(x)
        x = self.drop_out(x)
        out = self.out_proj(x)

        return out


class VariantTwoFineTuneClassifier(nn.Module):
    def __init__(self):
        super(VariantTwoFineTuneClassifier, self).__init__()
        self.HIDDEN_DIM = 768
        self.code_bert = RobertaModel.from_pretrained("microsoft/codebert-base", num_labels=2)
        self.classifier = VariantTwoClassifier()

    def forward(self, input_list_batch, mask_list_batch):
        d1, d2, d3 = input_list_batch.shape
        input_list_batch = torch.reshape(input_list_batch, (d1 * d2, d3))
        mask_list_batch = torch.reshape(mask_list_batch, (d1 * d2, d3))
        embeddings = self.code_bert(input_ids=input_list_batch, attention_mask=mask_list_batch).last_hidden_state[:, 0, :]
        embeddings = torch.reshape(embeddings, (d1, d2, self.HIDDEN_DIM))

        out = self.classifier(embeddings)

        return out


class VariantSixClassifier(nn.Module):
    def __init__(self):
        super(VariantSixClassifier, self).__init__()
        self.HIDDEN_DIM = 768
        self.HIDDEN_DIM_DROPOUT_PROB = 0.3

        self.before_linear = nn.Linear(5 * self.HIDDEN_DIM, self.HIDDEN_DIM)
        self.after_linear = nn.Linear(5 * self.HIDDEN_DIM, self.HIDDEN_DIM)
        self.combine = nn.Linear(2 * self.HIDDEN_DIM, self.HIDDEN_DIM)

        self.relu = nn.ReLU()
        self.drop_out = nn.Dropout(self.HIDDEN_DIM_DROPOUT_PROB)
        self.out_proj = nn.Linear(self.HIDDEN_DIM, 2)

    def forward(self, before_batch, after_batch):
        d1, d2, d3 = before_batch.shape
        before_batch = torch.reshape(before_batch, (d1, d2*d3))
        after_batch = torch.reshape(after_batch, (d1, d2*d3))

        before = self.before_linear(before_batch)
        after = self.after_linear(after_batch)
        combined = self.combine(torch.cat([before, after], axis=1))

        x = combined
        x = self.relu(x)
        x = self.drop_out(x)
        out = self.out_proj(x)

        return out


class VariantThreeClassifier(nn.Module):
    """An 1D Convulational Neural Network for Sentence Classification."""

    def __init__(self,
                 embed_dim=768,
                 filter_sizes=[3, 4, 5],
                 num_filters=[100, 100, 100],
                 num_classes=2,
                 dropout=0.5):
        super(VariantThreeClassifier, self).__init__()
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
        return out


class VariantSevenClassifier(nn.Module):
    """An 1D Convulational Neural Network for Sentence Classification."""

    def __init__(self,
                 embed_dim=768,
                 filter_sizes=[2, 3, 4],
                 num_filters=[100, 100, 100],
                 num_classes=2,
                 dropout=0.5):
        super(VariantSevenClassifier, self).__init__()
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
        self.fc = nn.Linear(2 * np.sum(num_filters), num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, before_batch, after_batch):
        """Perform a forward pass through the network.

        Args:
            input_ids (torch.Tensor): A tensor of token ids with shape
                (batch_size, max_sent_length)

        Returns:
            logits (torch.Tensor): Output logits with shape (batch_size,
                n_classes)
        """

        # Get embeddings from `input_ids`. Output shape: (b, max_len, embed_dim)
        x_embed_before = before_batch
        # batch, file, hidden_dim

        # Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
        # Output shape: (b, embed_dim, max_len)
        x_reshaped_before = x_embed_before.permute(0, 2, 1)

        # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
        x_conv_list_before = [F.relu(conv1d(x_reshaped_before)) for conv1d in self.conv1d_list]

        # Max pooling. Output shape: (b, num_filters[i], 1)
        x_pool_list_before = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
                       for x_conv in x_conv_list_before]

        # Concatenate x_pool_list to feed the fully connected layer.
        # Output shape: (b, sum(num_filters))
        x_fc_before = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list_before],
                         dim=1)

        # # Compute logits. Output shape: (b, n_classes)
        # out = self.fc(self.dropout(x_fc_before))


        ############################################


        x_embed_after = after_batch

        # Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
        # Output shape: (b, embed_dim, max_len)
        x_reshaped_after = x_embed_after.permute(0, 2, 1)

        # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
        x_conv_list_after = [F.relu(conv1d(x_reshaped_after)) for conv1d in self.conv1d_list]

        # Max pooling. Output shape: (b, num_filters[i], 1)
        x_pool_list_after = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
                       for x_conv in x_conv_list_after]

        # Concatenate x_pool_list to feed the fully connected layer.
        # Output shape: (b, sum(num_filters))
        x_fc_after = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list_after],
                         dim=1)

        # Compute logits. Output shape: (b, n_classes)

        x_fc = torch.cat([x_fc_before, x_fc_after], axis=1)
        out = self.fc(self.dropout(x_fc))

        return out


class VariantOneClassifier(nn.Module):
    def __init__(self):
        super(VariantOneClassifier, self).__init__()
        self.HIDDEN_DIM = 768
        self.DENSE_DIM = 128
        self.HIDDEN_DIM_DROPOUT_PROB = 0.5
        self.NUMBER_OF_LABELS = 2
        self.linear = nn.Linear(self.HIDDEN_DIM, self.DENSE_DIM)
        self.relu = nn.ReLU()

        self.drop_out = nn.Dropout(self.HIDDEN_DIM_DROPOUT_PROB)
        self.out_proj = nn.Linear(self.DENSE_DIM, self.NUMBER_OF_LABELS)

    def forward(self, embedding_batch):
        x = embedding_batch
        x = self.drop_out(x)
        x = self.linear(x)
        x = self.relu(x)
        x = self.drop_out(x)
        x = self.out_proj(x)

        return x


class VariantOneFinetuneClassifier(nn.Module):
    def __init__(self):
        super(VariantOneFinetuneClassifier, self).__init__()

        self.code_bert = RobertaModel.from_pretrained("microsoft/codebert-base", num_labels=2)
        self.classifier = VariantOneClassifier()

    def forward(self, input_batch, mask_batch):
        embeddings = self.code_bert(input_ids=input_batch, attention_mask=mask_batch)
        embeddings = embeddings.last_hidden_state[:, 0, :]
        out = self.classifier(embeddings)
        return out


class VariantFiveClassifier(nn.Module):
    def __init__(self):
        super(VariantFiveClassifier, self).__init__()
        self.HIDDEN_DIM = 768
        self.DENSE_DIM = 128
        self.HIDDEN_DIM_DROPOUT_PROB = 0.5
        self.NUMBER_OF_LABELS = 2
        self.linear = nn.Linear(2 * self.HIDDEN_DIM, self.DENSE_DIM)
        self.relu = nn.ReLU()

        self.drop_out = nn.Dropout(self.HIDDEN_DIM_DROPOUT_PROB)
        self.out_proj = nn.Linear(self.HIDDEN_DIM, self.NUMBER_OF_LABELS)

    def forward(self, before_batch, after_batch):
        combined = torch.cat([before_batch, after_batch], dim=1)
        x = combined
        x = self.drop_out(x)
        x = self.linear(x)
        x = self.relu(x)
        x = self.drop_out(x)
        x = self.out_proj(x)

        return x


class VariantFiveFineTuneClassifier(nn.Module):
    def __init__(self):
        super(VariantFiveFineTuneClassifier, self).__init__()
        self.code_bert = RobertaModel.from_pretrained("microsoft/codebert-base", num_labels=2)
        self.classifier = VariantFiveClassifier()

    def forward(self, added_input, added_mask, removed_input, removed_mask):
        added_embeddings = self.code_bert(input_ids=added_input, attention_mask=added_mask).last_hidden_state[:, 0, :]
        removed_embeddings = self.code_bert(input_ids=removed_input, attention_mask=removed_mask).last_hidden_state[:, 0, :]
        out = self.classifier(added_embeddings, removed_embeddings)
        return out


class VariantEightClassifier(nn.Module):
    def __init__(self):
        super(VariantEightClassifier, self).__init__()
        self.input_size = 768
        self.hidden_size = 128
        self.HIDDEN_DIM_DROPOUT_PROB = 0.3
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            batch_first=True,
                            bidirectional=True)
        self.linear = nn.Linear(4 * self.hidden_size, self.hidden_size)

        self.relu = nn.ReLU()

        self.drop_out = nn.Dropout(self.HIDDEN_DIM_DROPOUT_PROB)

        self.out_proj = nn.Linear(self.hidden_size, 2)

    def forward(self, before_batch, after_batch):
        self.lstm.flatten_parameters()
        before_out, (before_final_hidden_state, _) = self.lstm(before_batch)
        before_vector = before_out[:, 0]

        after_out, (after_final_hidden_state, _) = self.lstm(after_batch)
        after_vector = after_out[:, 0]

        x = self.linear(torch.cat([before_vector, after_vector], axis=1))

        x = self.relu(x)

        x = self.drop_out(x)

        out = self.out_proj(x)

        return out