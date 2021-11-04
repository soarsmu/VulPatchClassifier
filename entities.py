from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import os
import json
import torch

directory = os.path.dirname(os.path.abspath(__file__))


def get_average_value(embeddings):
    embeddings = torch.DoubleTensor(embeddings)
    sum_ = torch.sum(embeddings, dim=0)
    mean_ = torch.div(sum_, embeddings.shape[0])
    mean_ = mean_.detach()
    mean_ = mean_.cpu()

    return mean_


class PatchDataset(Dataset):
    def __init__(self, list_IDs, labels, id_to_url):
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

        before = get_average_value(data['before'])
        after = get_average_value(data['after'])
        y = self.labels[id]

        return int(id), url, before, after, y

