from transformers import RobertaTokenizer, RobertaConfig, RobertaModel, RobertaForSequenceClassification
import torch
from torch import nn as nn
import os
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torch import cuda
from sklearn import metrics
import numpy as np
import random
from transformers import AdamW
from transformers import get_scheduler
import csv
import pandas as pd
import sys
import math
is_test = False
test_size = 10000

directory = os.path.dirname(os.path.abspath(__file__))

dataset_name = 'ase_dataset_sept_19_2021.csv'

commit_code_folder_path = os.path.join(directory, 'commit_code')

model_folder_path = os.path.join(directory, 'model')


DATA_LOADER_PARAMS = {'batch_size': 128, 'shuffle': True, 'num_workers': 8}

LEARNING_RATE = 1e-5

use_cuda = cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

false_cases = []
CODE_LENGTH = 512
HIDDEN_DIM = 768

NEED_EXPANDED_VERSION = True

FIRST_PHASE_NUMBER_OF_EPOCHS = 6
NUMBER_OF_EPOCHS = 15

HIDDEN_DIM_DROPOUT_PROB = 0.3
NUMBER_OF_LABELS = 2
NUM_HEADS = 8

cos = nn.CosineSimilarity(dim=1)


def get_url_to_loc(item_list):
    url_to_loc = {}

    for item in item_list:
        url = item[0] + '/commit/' + item[1]
        loc = item[2]
        if url not in url_to_loc:
            url_to_loc[url] = 0
        url_to_loc[url] += loc

    return url_to_loc


def get_effort(df, lang, threshold, predicted):
    print("Calculating effort...")
    df = df[df.PL == lang]
    df_pos = df[df.label == 1]
    df_pos = df_pos[['repo', 'commit_id', 'LOC_MOD']]

    df_neg = df[df.label == 0]
    df_neg = df_neg[['repo', 'commit_id', 'LOC_MOD']]

    pos_items = df_pos.values.tolist()
    neg_items = df_neg.values.tolist()

    pos_url_to_loc = get_url_to_loc(pos_items)
    neg_url_to_loc = get_url_to_loc(neg_items)

    pos_loc_list = list(pos_url_to_loc.values())
    neg_loc_list = list(neg_url_to_loc.values())

    pos_loc_list.sort()
    neg_loc_list.sort()

    total_loc = 0
    for loc in pos_loc_list:
        total_loc += loc
    for loc in neg_loc_list:
        total_loc += loc

    count = 0
    total_vulnerabilities = 0
    for index, loc in enumerate(pos_loc_list):
        if (count + loc) / total_loc <= threshold:
            count += loc
        else:
            total_vulnerabilities = index - 1
            break

        if index == len(pos_loc_list) - 1 and (count + loc) / total_loc <= threshold:
            total_vulnerabilities = len(pos_loc_list)

    total_inspected = 0
    detected_vulnerabilities = 0
    predicted_indices = []
    non_vul_indices = []
    print("Total vulnerabilities: {}".format(total_vulnerabilities))
    commit_count = 0
    ifa = len(predicted)
    found_vuln = False
    for index, item in enumerate(predicted):
        commit_index = item[0]
        loc = item[2]
        label = item[3]
        rate = (total_inspected + loc) / total_loc
        if rate <= threshold:
            commit_count += 1
            total_inspected += loc
            if label == 1:
                if not found_vuln:
                    ifa = commit_count
                    found_vuln = True
                detected_vulnerabilities += 1
                predicted_indices.append(commit_index)
            else:
                non_vul_indices.append(commit_index)
        else:
            break
    recall = detected_vulnerabilities / total_vulnerabilities
    precision = detected_vulnerabilities / commit_count
    f1 = 2 * (precision * recall) / (precision + recall)
    pci = commit_count / len(predicted)

    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1: {}".format(f1))
    print("PIC: {}". format(pci))
    print("IFA: {}".format(ifa))

    return detected_vulnerabilities / total_vulnerabilities, predicted_indices, non_vul_indices


def get_recall_effort(threshold, predicted, total_vul):
    print("Calculating effort...")

    detected_vulnerabilities = 0
    predicted_indices = []
    non_vul_indices = []
    commit_count = 0
    total_commit = len(predicted)
    for index, item in enumerate(predicted):
        commit_index = item[0]
        label = item[3]
        rate = commit_count/total_commit
        if rate < threshold:
            commit_count += 1
            if label == 1:
                detected_vulnerabilities += 1
                predicted_indices.append(commit_index)
            else:
                non_vul_indices.append(commit_index)
        else:
            break

    return detected_vulnerabilities / total_vul, predicted_indices, non_vul_indices



def calculate_effort(predicted_path, lang, threshold):
    print("Reading dataset")
    df = pd.read_csv(dataset_name)
    df = df[df.partition == 'test']

    url_to_label, url_to_loc_mod = get_data()

    # row[0] : index => need to replace with url
    # row[1] : pred_prob

    items = []
    with open(predicted_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            items.append((row[0], float(row[1])))

    predicted = []
    url_to_pred = {}
    for item in items:
        predicted.append((item[0], item[1], url_to_loc_mod[item[0]], url_to_label[item[0]]))
        url_to_pred[item[0]] = item[1]
    predicted = sorted(predicted, key=lambda x: (-x[1], x[2]))
    effort_1, predicted_indices, non_vul_indices = get_effort(df, lang, threshold, predicted)
    # with open('vulcurator_non_vul.csv', 'w') as file:
    #     writer = csv.writer(file)
    #     for url in non_vul_indices:
    #         writer.writerow([url, url_to_loc_mod[url], url_to_pred[url]])

    # effort_2, predicted_indices_2 = get_effort(df, lang, 0.2, predicted)
    print("Effort {}%: {}".format(threshold, effort_1))
    # print("Effort 20%: {}".format(effort_2))

    return predicted_indices


def calculate_recall_effort(predicted_path, lang):
    url_to_label, url_to_loc_mod = get_data()

    total_vul = 300
    if lang == 'python':
        total_vul = 195

    print("Total vulnerabilities: {}".format(total_vul))

    # row[0] : index => need to replace with url
    # row[1] : pred_prob

    items = []
    with open(predicted_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            items.append((row[0], float(row[1])))

    predicted = []
    url_to_pred = {}
    for item in items:
        predicted.append((item[0], item[1], url_to_loc_mod[item[0]], url_to_label[item[0]]))
        url_to_pred[item[0]] = item[1]
    predicted = sorted(predicted, key=lambda x: (-x[1], x[2]))

    effort_1, predicted_indices, non_vul_indices = get_recall_effort(0.05, predicted, total_vul)

    effort_2, predicted_indices_2, non_vul_indices = get_recall_effort(0.2, predicted, total_vul)
    print("Effort 5%: {}".format(effort_1))
    print("Effort 20%: {}".format(effort_2))

    return predicted_indices

def get_normalized_effort(df, lang, threshold, predicted):
    print("Calculating effort...")
    df = df[df.PL == lang]
    df_pos = df[df.label == 1]
    df_pos = df_pos[['repo', 'commit_id', 'LOC_MOD']]

    df_neg = df[df.label == 0]
    df_neg = df_neg[['repo', 'commit_id', 'LOC_MOD']]

    pos_items = df_pos.values.tolist()
    neg_items = df_neg.values.tolist()

    pos_url_to_loc = get_url_to_loc(pos_items)
    neg_url_to_loc = get_url_to_loc(neg_items)

    pos_loc_list = list(pos_url_to_loc.values())
    neg_loc_list = list(neg_url_to_loc.values())

    pos_loc_list.sort()
    neg_loc_list.sort()

    total_loc = 0
    for loc in pos_loc_list:
        total_loc += loc
    for loc in neg_loc_list:
        total_loc += loc

    # Calculate AUC of optimal model

    count = 0
    x_optimal = []
    y_optimal = []
    x_optimal.append(0)
    y_optimal.append(0)

    for index, loc in enumerate(pos_loc_list):
        if (count + loc) / total_loc <= threshold:
            count += loc
            x_optimal.append(count / total_loc)
            y_optimal.append((index + 1) / len(pos_loc_list))
        else:
            break

    if (count + neg_loc_list[0]) / total_loc <= threshold:
        for loc in neg_loc_list:
            if (count + loc) / total_loc <= threshold:
                count += loc
                x_optimal.append(count / total_loc)
                y_optimal.append(1)

    auc_optimal = metrics.auc(x=x_optimal, y=y_optimal)

    # Calculate AUC of worst model

    count = 0
    x_worst = []
    y_worst = []
    x_worst.append(0)
    y_worst.append(0)
    neg_loc_list.reverse()
    pos_loc_list.reverse()

    for loc in neg_loc_list:
        if (count + loc) / total_loc <= threshold:
            count += 0
            x_worst.append(count / total_loc)
            y_worst.append(0)
        else:
            break

    if (count + pos_loc_list[0]) / total_loc <= threshold:
        for index, loc in enumerate(pos_loc_list):
            if (count + loc) / total_loc <= threshold:
                count += loc
                x_optimal.append(count / total_loc)
                y_optimal.append((index + 1) / len(pos_loc_list))

    auc_worst = metrics.auc(x=x_worst, y=y_worst)

    # Calculate AUC of model

    total_inspected = 0
    detected_vulnerabilities = 0
    x_model = []
    y_model = []
    x_model.append(0)
    y_model.append(0)
    for index, item in enumerate(predicted):
        loc = item[2]
        label = item[3]
        if (total_inspected + loc) / total_loc <= threshold:
            total_inspected += loc
            x_model.append(total_inspected / total_loc)
            if label == 1:
                detected_vulnerabilities += 1
            y_model.append(detected_vulnerabilities / len(pos_loc_list))
        else:
            break

    auc_model = metrics.auc(x=x_model, y=y_model)

    result = (auc_model - auc_worst) / (auc_optimal - auc_worst)

    return result


def get_data():
    print("Reading dataset...")
    df = pd.read_csv(dataset_name)
    df = df[['commit_id', 'repo', 'partition', 'diff', 'label', 'PL', 'LOC_MOD']]
    items = df.to_numpy().tolist()

    url_to_diff = {}
    url_to_label = {}
    url_to_pl = {}
    url_to_loc_mod = {}
    for item in items:
        commit_id = item[0]
        repo = item[1]
        url = repo + '/commit/' + commit_id
        label = item[4]
        pl = item[5]
        loc_mod = item[6]

        if url not in url_to_diff:
            url_to_diff[url] = ''
            url_to_loc_mod[url] = 0

        url_to_label[url] = label
        url_to_pl[url] = pl
        url_to_loc_mod[url] += loc_mod

    return url_to_label, url_to_loc_mod


def calculate_normalized_effort(predicted_path, lang):
    print("Reading dataset")
    df = pd.read_csv(dataset_name)
    df = df[df.partition == 'test']

    url_to_label, url_to_loc_mod = get_data()

    items = []
    with open(predicted_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            items.append((row[0], float(row[1])))

    predicted = []
    for item in items:
        predicted.append((item[0], item[1], url_to_loc_mod[item[0]], url_to_label[item[0]]))

    predicted = sorted(predicted, key=lambda x: (-x[1], x[2]))

    print("Normalized Effort 5%: {}".format(get_normalized_effort(df, lang, 0.05, predicted)))
    print("Normalized Effort 20%: {}".format(get_normalized_effort(df, lang, 0.2, predicted)))


def write_predicted_indices_to_file(result_file_path, predicted_indices_file_path):
    indices = calculate_effort(result_file_path, 'python')
    with open(predicted_indices_file_path, 'w') as writer:
        for index in indices:
            writer.write(str(index) + '\n')


# pure_classifier_java_file_path = 'huawei_pure_classifier_prob_java.txt'
# pure_classifier_python_file_path = 'huawei_pure_classifier_prob_python.txt'
# comparison_classifier_java_file_path = 'huawei_comparison_classifier_prob_java.txt'
# comparison_classifier_python_filepath = 'huawei_comparison_classifier_prob_python.txt'
# comparison_slice_java_file_path = 'huawei_comparison_slicing_classifier_prob_java.txt'
# comparison_slice_python_file_path = 'huawei_comparison_slicing_classifier_prob_python.txt'
# calculate_effort(comparison_slice_java_file_path, 'java')
# print('-' * 32)
# calculate_effort(comparison_classifier_python_filepath, 'python')
# print('-' * 32)
# calculate_normalized_effort(comparison_classifier_python_filepath, 'python')
# print('-' * 32)
# calculate_normalized_effort(comparison_slice_java_file_path, 'java')
# print('-' * 32)
# calculate_normalized_effort(comparison_slice_python_file_path, 'python')
# print('-' * 32)



# prob_python_path = 'probs/prob_variant_7_finetune_1_epoch_test_python.txt'
# calculate_effort(prob_python_path, 'python')
# print('-' * 32)


def test():
    df = pd.read_csv('test_dataset_predictions.csv')
    with open('probs/huawei_pred_prob_java.csv', 'w') as file:
        writer = csv.writer(file)
        for item in df.values.tolist():
            commit_id = item[0]
            repo = item[5]
            pred_prob = item[4]
            url = repo + '/commit/' + commit_id
            pl = item[7]
            if pl == 'java':
                writer.writerow([url, pred_prob])

# test()

# prob_java_path = 'probs/prob_ensemble_classifier_test_java.txt'
# calculate_effort(prob_java_path, 'java', 0.2)
#
# print('-' * 32)
# prob_python_path = 'probs/prob_ensemble_classifier_test_python.txt'
# calculate_effort(prob_python_path, 'python', 0.2)



def calculate_prob(prob, loc):
    return (prob * (1 + min(1, 1 / math.log(loc, 2)))) / 2


def write_new_metric(file_path, dest_path):
    df = pd.read_csv(file_path, header=None)
    url_to_label, url_to_loc_mod = get_data()
    with open(dest_path, 'w') as file:
        writer = csv.writer(file)
        for item in df.values.tolist():
            url = item[0]
            prob = item[1]
            loc = url_to_loc_mod[url]
            new_prob = calculate_prob(prob, loc)
            writer.writerow([url, new_prob])


def calculate_auc(prob_path, url_to_label):
    df = pd.read_csv(prob_path, header=None)

    y_test = []
    y_pred = []
    for item in df.values.tolist():
        url = item[0]
        prob = item[1]
        label = url_to_label[url]
        y_test.append(label)
        y_pred.append(prob)

    auc = metrics.roc_auc_score(y_true=y_test, y_score=y_pred)

    return auc

def test_new_metric():
    huawei_prob_path_java = 'probs/huawei_pred_prob_java.csv'
    huawei_prob_path_python = 'probs/huawei_pred_prob_python.csv'
    model_prob_path_java = 'probs/prob_ensemble_classifier_test_java.txt'
    model_prob_path_python = 'probs/prob_ensemble_classifier_test_python.txt'
    huawei_new_prob_java_path = 'probs/huawei_new_prob_java.txt'
    huawei_new_prob_python_path = 'probs/huawei_new_prob_python.txt'
    model_new_prob_java_path = 'probs/new_prob_java.txt'
    model_new_prob_python_path = 'probs/new_prob_python.txt'

    # write_new_metric(huawei_prob_path_java, huawei_new_prob_java_path)
    # write_new_metric(huawei_prob_path_python, huawei_new_prob_python_path)
    write_new_metric(model_prob_path_java, model_new_prob_java_path)
    write_new_metric(model_prob_path_python, model_new_prob_python_path)

    url_to_label, url_to_loc_mod = get_data()

    # huawei_java_auc = calculate_auc(huawei_new_prob_java_path, url_to_label)
    # print("Huawei java auc: {}".format(huawei_java_auc))
    #
    # huawei_python_auc = calculate_auc(huawei_new_prob_python_path, url_to_label)
    # print("Huawei python auc: {}".format(huawei_python_auc))

    model_java_auc = calculate_auc(model_new_prob_java_path, url_to_label)
    print("Model java auc: {}".format(model_java_auc))

    model_python_auc = calculate_auc(model_new_prob_python_path, url_to_label)
    print("Model python auc: {}".format(model_python_auc))

    calculate_effort(model_new_prob_java_path, 'java', 0.05)
    calculate_effort(model_new_prob_java_path, 'java', 0.2)
    calculate_normalized_effort(model_new_prob_java_path, 'java')

    calculate_effort(model_new_prob_python_path, 'python', 0.05)
    calculate_effort(model_new_prob_python_path, 'python', 0.2)
    calculate_normalized_effort(model_new_prob_python_path, 'python')


test_new_metric()