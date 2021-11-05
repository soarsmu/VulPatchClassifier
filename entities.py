from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import RobertaTokenizer, RobertaModel
import os
import json
import torch

directory = os.path.dirname(os.path.abspath(__file__))
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
code_bert = RobertaModel.from_pretrained("microsoft/codebert-base", num_labels=2)
empty_code = tokenizer.sep_token + ''
inputs = tokenizer([empty_code], padding=True, max_length=512, truncation=True, return_tensors="pt")
input_ids, attention_mask = inputs.data['input_ids'][0], inputs.data['attention_mask'][0]
empty_embedding = code_bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :].tolist()


# def get_average_value(embeddings):
#     embeddings = torch.FloatTensor(embeddings)
#     sum_ = torch.sum(embeddings, dim=0)
#     mean_ = torch.div(sum_, embeddings.shape[0])
#     mean_ = mean_.detach()
#     mean_ = mean_.cpu()
#
#     return mean_


class PatchDataset(Dataset):
    def __init__(self, list_IDs, labels, id_to_url):
        self.max_data_length = 5
        self.list_IDs = list_IDs
        self.labels = labels
        self.id_to_url = id_to_url

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        id = self.list_IDs[index]
        url = self.id_to_url[id]
        file_path = os.path.join(directory, '../file_data/' + url.replace('/', '_') + '.txt')

        with open(file_path, 'r') as reader:
            data = json.loads(reader.read())

        before = data['before']
        after = data['after']
        if len(before) > 5:
            before = before[:5]
        if len(after) > 5:
            after = after[:5]
        while len(before) < 5:
            before.append(empty_embedding)
        while len(after) < 5:
            after.append(empty_embedding)

        before = torch.FloatTensor(before)
        after = torch.FloatTensor(after)

        y = self.labels[id]

        return int(id), url, before, after, y

