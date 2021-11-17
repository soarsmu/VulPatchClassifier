from transformers import RobertaTokenizer, RobertaModel
import pandas as pd
import json
import os
import torch
import tqdm
from torch import cuda
from torch import nn as nn
import matplotlib.pyplot as plt


EMBEDDING_DIRECTORY = '../embeddings/variant_3'

directory = os.path.dirname(os.path.abspath(__file__))

dataset_name = 'ase_dataset_sept_19_2021.csv'
# dataset_name = 'huawei_sub_dataset_new.csv'

CODE_LINE_LENGTH = 256

use_cuda = cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
random_seed = 109
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


def get_input_and_mask(tokenizer, code_list):
    inputs = tokenizer(code_list, padding=True, max_length=CODE_LINE_LENGTH, truncation=True, return_tensors="pt")

    return inputs.data['input_ids'], inputs.data['attention_mask']


def get_code_version(diff, added_version):
    code = ''
    lines = diff.splitlines()
    for line in lines:
        mark = '+'
        if not added_version:
            mark = '-'
        if line.startswith(mark):
            line = line[1:].strip()
            if line.startswith(('//', '/**', '/*', '*', '*/', '#')):
                continue
            code = code + line + '\n'

    return code


def get_hunk_embeddings(code_list, tokenizer, code_bert):
    # process all lines in one
    input_ids, attention_mask = get_input_and_mask(tokenizer, code_list)

    with torch.no_grad():
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        embeddings = code_bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        input_ids.to('cpu')
        attention_mask.to('cpu')
        embeddings.to('cpu')
    embeddings = embeddings.tolist()
    return embeddings


def write_embeddings_to_files(code_list, url_list, tokenizer, code_bert):
    hunk_embeddings = get_hunk_embeddings(code_list, tokenizer, code_bert)
    url_to_embeddings = {}
    for index, url in enumerate(url_list):
        if url not in url_to_embeddings:
            url_to_embeddings[url] = []
        url_to_embeddings[url].append(hunk_embeddings[index])

    url_to_data = {}
    for url, embeddings in url_to_embeddings.items():
        data = {'embeddings': embeddings}
        url_to_data[url] = data
    for url, data in url_to_data.items():
        file_path = os.path.join(directory, EMBEDDING_DIRECTORY + '/' + url.replace('/', '_') + '.txt')
        json.dump(data, open(file_path, 'w'))


def hunk_empty(hunk):
    return hunk == ''


def get_hunk_from_diff(diff):
    hunk_list = []
    hunk = ''
    for line in diff.split('\n'):
        if line.startswith(('+', '-')):
            hunk = hunk + line + '\n'
        else:
            if not hunk_empty(hunk):    # finish a hunk
                hunk = hunk[:-1]
                hunk_list.append(hunk)
                hunk = ''

    if not hunk_empty(hunk):
        hunk_list.append(hunk)

    return hunk_list


def get_data():
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    code_bert = RobertaModel.from_pretrained("microsoft/codebert-base", num_labels=2)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        code_bert = nn.DataParallel(code_bert)

    code_bert.to(device)
    code_bert.eval()
    print("Reading dataset...")
    df = pd.read_csv(dataset_name)
    df = df[['commit_id', 'repo', 'partition', 'diff', 'label', 'PL', 'LOC_MOD', 'filename']]
    items = df.to_numpy().tolist()

    url_to_hunk = {}

    for item in items:
        commit_id = item[0]
        repo = item[1]
        url = repo + '/commit/' + commit_id
        diff = item[3]

        if url not in url_to_hunk:
            url_to_hunk[url] = []

        url_to_hunk[url].extend(get_hunk_from_diff(diff))

    code_list = []
    url_list = []
    for url, diff_list in tqdm.tqdm(url_to_hunk.items()):
        for i, diff in enumerate(diff_list):
            removed_code = get_code_version(diff, False)
            added_code = get_code_version(diff, True)

            code = removed_code + tokenizer.sep_token + added_code

            code_list.append(code)
            url_list.append(url)

        if len(url_list) >= 50:
            write_embeddings_to_files(code_list, url_list, tokenizer, code_bert)
            code_list = []
            url_list = []

    write_embeddings_to_files(code_list, url_list, tokenizer, code_bert)


def get_token_count(code, tokenizer):
    inputs = tokenizer(code, return_tensors="pt")
    token_count = inputs.data['input_ids'].shape[1]

    return token_count


def get_hunk_token_count_distribution():
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    code_bert = RobertaModel.from_pretrained("microsoft/codebert-base", num_labels=2)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        code_bert = nn.DataParallel(code_bert)

    code_bert.to(device)
    code_bert.eval()
    print("Reading dataset...")
    df = pd.read_csv(dataset_name)
    df = df[df.label == 1]
    df = df[['diff']]

    hunk_list = []
    for item in df.values.tolist():
        diff = item[0]
        hunk_list.extend(get_hunk_from_diff(diff))

    added_token_count = []
    removed_token_count = []
    for hunk in tqdm.tqdm(hunk_list):
        removed_code = tokenizer.sep_token + get_code_version(hunk, False)
        added_code = tokenizer.sep_token + get_code_version(hunk, True)
        added_token_count.append(get_token_count(added_code, tokenizer))
        removed_token_count.append(get_token_count(removed_code, tokenizer))

    plt.boxplot([added_token_count, removed_token_count],
                labels=['Added_token_count', "Removed_token_count"], showfliers=True)
    plt.title("Distribution in token count by hunk")
    plt.ylabel("No. Token count")
    plt.show()


if __name__ == '__main__':
    get_data()
