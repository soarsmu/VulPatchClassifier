from transformers import RobertaTokenizer, RobertaModel
import pandas as pd
import json
import os
import torch
from tqdm import tqdm
from torch import cuda
from torch import nn as nn
import matplotlib.pyplot as plt

directory = os.path.dirname(os.path.abspath(__file__))

dataset_name = 'ase_dataset_sept_19_2021.csv'
# dataset_name = 'huawei_sub_dataset_new.csv'

use_cuda = cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
random_seed = 109
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

DATASET_FILE_FOLDER_PATH = os.path.join(directory, '../dataset_files')
DATASET_REMOVED_FILE_FOLDER_PATH = os.path.join(directory, '../dataset_removed_files')


def get_file_to_manual_map():
    df = pd.read_csv(dataset_name)
    df = df[['commit_id', 'repo']]
    for index, item in tqdm(enumerate(df.values.tolist())):
        source_file_path = DATASET_FILE_FOLDER_PATH + '/' + str(index) + '.txt'
        removed_file_file_path = DATASET_REMOVED_FILE_FOLDER_PATH + '/' + str(index) + '.txt'

        if not os.path.isfile(source_file_path) and not os.path.isfile(removed_file_file_path):
            print("https://github.com/" + item[1] + '/commit/' + item[0])


get_file_to_manual_map()