import torch
from torch import nn as nn
import os
from torch.utils.data import Dataset, DataLoader
from torch import cuda
import pandas as pd
from entities import VariantOneDataset, VariantTwoDataset, VariantFiveDataset, VariantSixDataset, VariantThreeDataset, VariantSevenDataset
from model import VariantOneClassifier, VariantTwoClassifier, VariantFiveClassifier, VariantSixClassifier, VariantThreeClassifier, VariantSevenClassifier
import utils
import variant_1
import variant_2
import variant_3
import variant_5
import variant_6
import variant_7
from sklearn import metrics
from statistics import mean
from sklearn.linear_model import LogisticRegression
import csv
# dataset_name = 'huawei_csv_subset_slicing_limited_10.csv'
# dataset_name = 'huawei_sub_dataset.csv'
dataset_name = 'ase_dataset_sept_19_2021.csv'

directory = os.path.dirname(os.path.abspath(__file__))

model_folder_path = os.path.join(directory, 'model')

VARIANT_ONE_MODEL_PATH = 'model/patch_variant_1_best_model.sav'
VARIANT_TWO_MODEL_PATH = 'model/patch_variant_2_best_model.sav'
VARIANT_THREE_MODEL_PATH = 'model/patch_variant_3_best_model.sav'
VARIANT_FIVE_MODEL_PATH = 'model/patch_variant_5_best_model.sav'
VARIANT_SIX_MODEL_PATH = 'model/patch_variant_6_best_model.sav'
VARIANT_SEVEN_MODEL_PATH = 'model/patch_variant_7_best_model.sav'


TEST_BATCH_SIZE = 128

TEST_PARAMS = {'batch_size': TEST_BATCH_SIZE, 'shuffle': True, 'num_workers': 8}
use_cuda = cuda.is_available()
device = torch.device("cuda:1" if use_cuda else "cpu")
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


def infer_variant_1(partition, result_file_path):
    print("Testing on partition: {}".format(partition))
    print("Saving result to: {}".format(result_file_path))

    print("Testing...")

    model = VariantOneClassifier()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.load_state_dict(torch.load(VARIANT_ONE_MODEL_PATH, map_location='cuda:1'))
    model.to(device)

    ids, id_to_label, id_to_url = get_dataset_info(partition)
    dataset = VariantOneDataset(ids, id_to_label, id_to_url)
    generator = DataLoader(dataset, **TEST_PARAMS)

    precision, recall, f1, auc, urls, probs = variant_1.predict_test_data(model, generator, device, need_prob=True)

    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1: {}".format(f1))
    print("AUC: {}".format(auc))
    print("-" * 32)

    write_prob_to_file(result_file_path, urls, probs)


def infer_variant_2(partition, result_file_path):
    print("Testing on partition: {}".format(partition))
    print("Saving result to: {}".format(result_file_path))

    print("Testing...")

    model = VariantTwoClassifier()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)

    model.load_state_dict(torch.load(VARIANT_TWO_MODEL_PATH))

    ids, id_to_label, id_to_url = get_dataset_info(partition)
    dataset = VariantTwoDataset(ids, id_to_label, id_to_url)
    generator = DataLoader(dataset, **TEST_PARAMS)

    precision, recall, f1, auc, urls, probs = variant_2.predict_test_data(model, generator, device, need_prob=True)

    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1: {}".format(f1))
    print("AUC: {}".format(auc))
    print("-" * 32)

    write_prob_to_file(result_file_path, urls, probs)


def infer_variant_3(partition, result_file_path):
    print("Testing on partition: {}".format(partition))
    print("Saving result to: {}".format(result_file_path))

    print("Testing...")

    model = VariantThreeClassifier()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)

    model.load_state_dict(torch.load(VARIANT_THREE_MODEL_PATH))

    ids, id_to_label, id_to_url = get_dataset_info(partition)
    dataset = VariantThreeDataset(ids, id_to_label, id_to_url)
    generator = DataLoader(dataset, **TEST_PARAMS)

    precision, recall, f1, auc, urls, probs = variant_3.predict_test_data(model, generator, device, need_prob=True)

    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1: {}".format(f1))
    print("AUC: {}".format(auc))
    print("-" * 32)

    write_prob_to_file(result_file_path, urls, probs)


def infer_variant_5(partition, result_file_path):
    print("Testing on partition: {}".format(partition))
    print("Saving result to: {}".format(result_file_path))

    print("Testing...")

    model = VariantFiveClassifier()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)

    model.load_state_dict(torch.load(VARIANT_FIVE_MODEL_PATH))

    ids, id_to_label, id_to_url = get_dataset_info(partition)
    dataset = VariantFiveDataset(ids, id_to_label, id_to_url)
    generator = DataLoader(dataset, **TEST_PARAMS)

    precision, recall, f1, auc, urls, probs = variant_5.predict_test_data(model, generator, device, need_prob=True)

    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1: {}".format(f1))
    print("AUC: {}".format(auc))
    print("-" * 32)

    write_prob_to_file(result_file_path, urls, probs)


def infer_variant_6(partition, result_file_path):
    print("Testing on partition: {}".format(partition))
    print("Saving result to: {}".format(result_file_path))

    print("Testing...")

    model = VariantSixClassifier()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)

    model.load_state_dict(torch.load(VARIANT_SIX_MODEL_PATH))

    ids, id_to_label, id_to_url = get_dataset_info(partition)
    dataset = VariantSixDataset(ids, id_to_label, id_to_url)
    generator = DataLoader(dataset, **TEST_PARAMS)

    precision, recall, f1, auc, urls, probs = variant_6.predict_test_data(model, generator, device, need_prob=True)

    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1: {}".format(f1))
    print("AUC: {}".format(auc))
    print("-" * 32)

    write_prob_to_file(result_file_path, urls, probs)


def infer_variant_7(partition, result_file_path):
    print("Testing on partition: {}".format(partition))
    print("Saving result to: {}".format(result_file_path))

    print("Testing...")

    model = VariantSevenClassifier()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)

    model.load_state_dict(torch.load(VARIANT_SEVEN_MODEL_PATH))

    ids, id_to_label, id_to_url = get_dataset_info(partition)
    dataset = VariantSevenDataset(ids, id_to_label, id_to_url)
    generator = DataLoader(dataset, **TEST_PARAMS)

    precision, recall, f1, auc, urls, probs = variant_7.predict_test_data(model, generator, device, need_prob=True)

    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1: {}".format(f1))
    print("AUC: {}".format(auc))
    print("-" * 32)

    write_prob_to_file(result_file_path, urls, probs)


def get_dataset_info(partition):
    url_data, label_data = utils.get_data(dataset_name)
    ids = []
    index = 0
    id_to_url = {}
    id_to_label = {}

    for i, url in enumerate(url_data[partition]):
        ids.append(index)
        id_to_url[index] = url
        id_to_label[index] = label_data[partition][i]
        index += 1

    return ids, id_to_label, id_to_url


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


def get_data_ensemble_model(prob_list, label_list):
    clf = LogisticRegression(random_state=109).fit(prob_list, label_list)

    return clf


def get_variant_result(java_result_path, python_result_path):
    result_java = read_pred_prob(java_result_path)
    result_python = read_pred_prob(python_result_path)
    result = {}
    result.update(result_java)
    result.update(result_python)

    return result


def get_prob(result_list, url):
    return [result[url] for result in result_list]


def get_partition_prob_list(result_path_list, partition):
    result_list = []
    for result_path in result_path_list:
        variant_result = get_variant_result(result_path[0], result_path[1])
        result_list.append(variant_result)

    url_data, label_data = utils.get_data(dataset_name)

    prob_list, label_list = [], []

    for i, url in enumerate(url_data[partition]):
        prob_list.append(get_prob(result_list, url))
        label_list.append(url_data[partition][i])

    return prob_list, label_list


def get_combined_ensemble_model():
    train_result_path_list = [
        ['variant_1_prob_train_java.txt', 'variant_1_prob_train_python.txt'],
        ['variant_2_prob_train_java.txt', 'variant_2_prob_train_python.txt'],
        ['variant_3_prob_train_java.txt', 'variant_3_prob_train_python.txt'],
        ['variant_5_prob_train_java.txt', 'variant_5_prob_train_python.txt'],
        ['variant_6_prob_train_java.txt', 'variant_6_prob_train_python.txt'],
        ['variant_7_prob_train_java.txt', 'variant_7_prob_train_python.txt']
    ]

    val_result_path_list = [
        ['variant_1_prob_val_java.txt', 'variant_1_prob_val_python.txt'],
        ['variant_2_prob_val_java.txt', 'variant_2_prob_val_python.txt'],
        ['variant_3_prob_val_java.txt', 'variant_3_prob_val_python.txt'],
        ['variant_5_prob_val_java.txt', 'variant_5_prob_val_python.txt'],
        ['variant_6_prob_val_java.txt', 'variant_6_prob_val_python.txt'],
        ['variant_7_prob_val_java.txt', 'variant_7_prob_val_python.txt']
    ]

    test_result_path_list = [
        ['variant_1_prob_java.txt', 'variant_1_prob_python.txt'],
        ['variant_2_prob_java.txt', 'variant_2_prob_python.txt'],
        ['variant_3_prob_java.txt', 'variant_3_prob_python.txt'],
        ['variant_5_prob_java.txt', 'variant_5_prob_python.txt'],
        ['variant_6_prob_java.txt', 'variant_6_prob_python.txt'],
        ['variant_7_prob_java.txt', 'variant_7_prob_python.txt'],
    ]

    train_prob_list, train_label_list = get_partition_prob_list(train_result_path_list, 'train')
    val_prob_list, val_label_list = get_partition_prob_list(val_result_path_list, 'val')
    test_prob_list, test_label_list = get_partition_prob_list(test_result_path_list, 'test')

    train_ensemble_model = get_data_ensemble_model(train_prob_list, train_label_list)
    val_ensemble_model = get_data_ensemble_model(val_prob_list, val_label_list)


infer_variant_1('val', 'variant_1_prob_val.txt')

