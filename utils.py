import pandas as pd
from tqdm import tqdm
import os.path
import json


def get_data_from_saved_file(file_info_name):
    with open(file_info_name, 'r') as reader:
        data = json.loads(reader.read())

    return data['url_data'], data['label_data']


def get_data(dataset_name):
    file_info_name = 'info_' + dataset_name + '.json'
    if os.path.isfile(file_info_name):
        return get_data_from_saved_file(file_info_name)

    print("Reading dataset...")
    df = pd.read_csv(dataset_name)
    df = df[['commit_id', 'repo', 'partition', 'PL', 'label']]
    items = df.to_numpy().tolist()

    url_train, url_val_java, url_val_python, url_test_java, url_test_python = [], [], [], [], []
    label_train, label_val_java, label_val_python, label_test_java, label_test_python = [], [], [], [], []

    for item in tqdm(items):
        commit_id = item[0]
        repo = item[1]
        url = repo + '/commit/' + commit_id
        partition = item[2]
        pl = item[3]
        label = item[4]

        if partition == 'train':
            if url not in url_train:
                url_train.append(url)
                label_train.append(label)
        elif partition == 'val':
            if pl == 'java' and url not in url_val_java:
                url_val_java.append(url)
                label_val_java.append(label)
            elif pl == 'python' and url not in url_val_python:
                url_val_python.append(url)
                label_val_python.append(label)
        elif partition == 'test':
            if pl == 'java' and url not in url_test_java:
                url_test_java.append(url)
                label_test_java.append(label)
            elif pl == 'python' and url not in url_test_python:
                url_test_python.append(url)
                label_test_python.append(label)
        else:
            Exception("Invalid partition: {}".format(partition))

    print("Finish reading dataset")
    url_data = {'train': url_train, 'val_java': url_val_java, 'val_python': url_val_python,
                'test_java': url_test_java, 'test_python': url_test_python}
    label_data = {'train': label_train, 'val_java': label_val_java, 'val_python': label_val_python,
                'test_java': label_test_java, 'test_python': label_test_python}

    data = {'url_data': url_data, 'label_data': label_data}

    json.dump(data, open(file_info_name, 'w'))

    return url_data, label_data
