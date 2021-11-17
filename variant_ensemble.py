import torch
from torch import nn as nn
import os
from torch.utils.data import Dataset, DataLoader
from torch import cuda
import pandas as pd
from entities import VariantOneDataset, VariantTwoDataset, VariantFiveDataset, VariantSixDataset
from model import VariantOneClassifier, VariantTwoClassifier, VariantFiveClassifier, VariantSixClassifier
import utils
import variant_1
import variant_2
import variant_5
import variant_6
from sklearn import metrics
from statistics import mean
import csv
# dataset_name = 'huawei_csv_subset_slicing_limited_10.csv'
# dataset_name = 'huawei_sub_dataset.csv'
dataset_name = 'ase_dataset_sept_19_2021.csv'

directory = os.path.dirname(os.path.abspath(__file__))

model_folder_path = os.path.join(directory, 'model')

VARIANT_ONE_MODEL_PATH = 'model/patch_variant_1_best_model.sav'
VARIANT_TWO_MODEL_PATH = 'model/patch_variant_2_best_model.sav'
VARIANT_FIVE_MODEL_PATH = 'model/patch_variant_5_best_model.sav'
VARIANT_SIX_MODEL_PATH = 'model/patch_variant_6_best_model.sav'


TEST_BATCH_SIZE = 128

TEST_PARAMS = {'batch_size': TEST_BATCH_SIZE, 'shuffle': True, 'num_workers': 8}
use_cuda = cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
random_seed = 109
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


def write_prob_to_file(file_path, urls, probs):
    with open(file_path, 'w') as file:
        writer = csv.writer(file)
        for i, url in enumerate(urls):
            writer.writerow([url, probs[i]])


def get_variant_one_result():
    model = VariantOneClassifier()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)

    model.load_state_dict(torch.load(VARIANT_ONE_MODEL_PATH))

    url_data, label_data = utils.get_data(dataset_name)

    train_ids, val_ids, test_java_ids, test_python_ids = [], [], [], []
    index = 0
    id_to_url = {}
    id_to_label = {}

    for i, url in enumerate(url_data['test_java']):
        test_java_ids.append(index)
        id_to_url[index] = url
        id_to_label[index] = label_data['test_java'][i]
        index += 1

    for i, url in enumerate(url_data['test_python']):
        test_python_ids.append(index)
        id_to_url[index] = url
        id_to_label[index] = label_data['test_python'][i]
        index += 1

    test_java_set = VariantOneDataset(test_java_ids, id_to_label, id_to_url)
    test_python_set = VariantOneDataset(test_python_ids, id_to_label, id_to_url)
    test_java_generator = DataLoader(test_java_set, **TEST_PARAMS)
    test_python_generator = DataLoader(test_python_set, **TEST_PARAMS)

    print("Testing on Java...")
    precision, recall, f1, auc, urls, probs = variant_1.predict_test_data(model, test_java_generator, device, need_prob=True)

    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1: {}".format(f1))
    print("AUC: {}".format(auc))
    print("-" * 32)

    write_prob_to_file("variant_1_prob_java.txt", urls, probs)

    print("Result on Python testing dataset...")
    precision, recall, f1, auc, urls, probs = variant_1.predict_test_data(model=model,
                                                   testing_generator=test_python_generator,
                                                   device=device, need_prob=True)

    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1: {}".format(f1))
    print("AUC: {}".format(auc))
    print("-" * 32)

    write_prob_to_file("variant_1_prob_python.txt", urls, probs)


def get_variant_two_result():
    model = VariantTwoClassifier()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)

    model.load_state_dict(torch.load(VARIANT_TWO_MODEL_PATH))

    url_data, label_data = utils.get_data(dataset_name)

    train_ids, val_ids, test_java_ids, test_python_ids = [], [], [], []
    index = 0
    id_to_url = {}
    id_to_label = {}

    for i, url in enumerate(url_data['test_java']):
        test_java_ids.append(index)
        id_to_url[index] = url
        id_to_label[index] = label_data['test_java'][i]
        index += 1

    for i, url in enumerate(url_data['test_python']):
        test_python_ids.append(index)
        id_to_url[index] = url
        id_to_label[index] = label_data['test_python'][i]
        index += 1

    test_java_set = VariantTwoDataset(test_java_ids, id_to_label, id_to_url)
    test_python_set = VariantTwoDataset(test_python_ids, id_to_label, id_to_url)
    test_java_generator = DataLoader(test_java_set, **TEST_PARAMS)
    test_python_generator = DataLoader(test_python_set, **TEST_PARAMS)

    print("Testing on Java...")
    precision, recall, f1, auc, urls, probs = variant_2.predict_test_data(model, test_java_generator, device,
                                                                          need_prob=True)

    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1: {}".format(f1))
    print("AUC: {}".format(auc))
    print("-" * 32)

    write_prob_to_file("variant_2_prob_java.txt", urls, probs)

    print("Result on Python testing dataset...")
    precision, recall, f1, auc, urls, probs = variant_2.predict_test_data(model=model,
                                                                          testing_generator=test_python_generator,
                                                                          device=device, need_prob=True)

    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1: {}".format(f1))
    print("AUC: {}".format(auc))
    print("-" * 32)

    write_prob_to_file("variant_2_prob_python.txt", urls, probs)


def get_variant_five_result():
    model = VariantFiveClassifier()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)

    model.load_state_dict(torch.load(VARIANT_FIVE_MODEL_PATH))

    url_data, label_data = utils.get_data(dataset_name)

    train_ids, val_ids, test_java_ids, test_python_ids = [], [], [], []
    index = 0
    id_to_url = {}
    id_to_label = {}

    for i, url in enumerate(url_data['test_java']):
        test_java_ids.append(index)
        id_to_url[index] = url
        id_to_label[index] = label_data['test_java'][i]
        index += 1

    for i, url in enumerate(url_data['test_python']):
        test_python_ids.append(index)
        id_to_url[index] = url
        id_to_label[index] = label_data['test_python'][i]
        index += 1

    test_java_set = VariantFiveDataset(test_java_ids, id_to_label, id_to_url)
    test_python_set = VariantFiveDataset(test_python_ids, id_to_label, id_to_url)
    test_java_generator = DataLoader(test_java_set, **TEST_PARAMS)
    test_python_generator = DataLoader(test_python_set, **TEST_PARAMS)

    print("Testing on Java...")
    precision, recall, f1, auc, urls, probs = variant_5.predict_test_data(model, test_java_generator, device,
                                                                          need_prob=True)

    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1: {}".format(f1))
    print("AUC: {}".format(auc))
    print("-" * 32)

    write_prob_to_file("variant_5_prob_java.txt", urls, probs)

    print("Result on Python testing dataset...")
    precision, recall, f1, auc, urls, probs = variant_5.predict_test_data(model=model,
                                                                          testing_generator=test_python_generator,
                                                                          device=device, need_prob=True)

    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1: {}".format(f1))
    print("AUC: {}".format(auc))
    print("-" * 32)

    write_prob_to_file("variant_5_prob_python.txt", urls, probs)


def get_variant_six_result():
    model = VariantSixClassifier()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)

    model.load_state_dict(torch.load(VARIANT_SIX_MODEL_PATH))

    url_data, label_data = utils.get_data(dataset_name)

    train_ids, val_ids, test_java_ids, test_python_ids = [], [], [], []
    index = 0
    id_to_url = {}
    id_to_label = {}

    for i, url in enumerate(url_data['test_java']):
        test_java_ids.append(index)
        id_to_url[index] = url
        id_to_label[index] = label_data['test_java'][i]
        index += 1

    for i, url in enumerate(url_data['test_python']):
        test_python_ids.append(index)
        id_to_url[index] = url
        id_to_label[index] = label_data['test_python'][i]
        index += 1

    test_java_set = VariantSixDataset(test_java_ids, id_to_label, id_to_url)
    test_python_set = VariantSixDataset(test_python_ids, id_to_label, id_to_url)
    test_java_generator = DataLoader(test_java_set, **TEST_PARAMS)
    test_python_generator = DataLoader(test_python_set, **TEST_PARAMS)

    print("Testing on Java...")
    precision, recall, f1, auc, urls, probs = variant_6.predict_test_data(model, test_java_generator, device,
                                                                          need_prob=True)

    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1: {}".format(f1))
    print("AUC: {}".format(auc))
    print("-" * 32)

    write_prob_to_file("variant_6_prob_java.txt", urls, probs)

    print("Result on Python testing dataset...")
    precision, recall, f1, auc, urls, probs = variant_6.predict_test_data(model=model,
                                                                          testing_generator=test_python_generator,
                                                                          device=device, need_prob=True)

    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1: {}".format(f1))
    print("AUC: {}".format(auc))
    print("-" * 32)

    write_prob_to_file("variant_6_prob_python.txt", urls, probs)


def read_pred_prob(file_path):
    df = pd.read_csv(file_path, header=None)
    url_to_prob = {}

    for url, prob in df.values.tolist():
        url_to_prob[url] = prob

    return url_to_prob


def get_auc_max_ensemble():
    print("Reading result...")
    variant_1_result = read_pred_prob('variant_1_prob_java.txt')
    variant_2_result = read_pred_prob('variant_2_prob_java.txt')
    variant_5_result = read_pred_prob('variant_5_prob_java.txt')
    variant_6_result = read_pred_prob('variant_6_prob_java.txt')

    print("Finish reading result")

    url_to_max_prob = {}

    for url, prob_1 in variant_1_result.items():
        prob_2 = variant_2_result[url]
        prob_5 = variant_5_result[url]
        prob_6 = variant_6_result[url]
        url_to_max_prob[url] = mean([prob_1, prob_2, prob_5, prob_6])

    url_data, label_data = utils.get_data(dataset_name)
    url_test = url_data['test_java']
    label_test = label_data['test_java']

    y_score = []
    y_true = []
    for i, url in enumerate(url_test):
        y_true.append(label_test[i])
        y_score.append(url_to_max_prob[url])

    auc = metrics.roc_auc_score(y_true=y_true, y_score=y_score)

    print("AUC: {}".format(auc))


get_auc_max_ensemble()