from transformers import RobertaTokenizer, RobertaModel
import torch
from torch import cuda
from torch import nn as nn
import preprocess_finetuned_variant_1, preprocess_finetuned_variant_2, preprocess_finetuned_variant_3, preprocess_finetuned_variant_5, preprocess_finetuned_variant_6, preprocess_finetuned_variant_7, preprocess_finetuned_variant_8
import pandas as pd
from torch.nn import functional as F
import adjustment_runner
import utils
import ensemble_classifier
from entities import EnsembleDataset
from torch.utils.data import DataLoader

from model import VariantOneFinetuneClassifier, VariantTwoFineTuneClassifier, VariantOneClassifier, VariantTwoClassifier, VariantFiveClassifier, VariantSixClassifier, \
    VariantThreeClassifier, VariantSevenClassifier, VariantEightClassifier, VariantThreeFineTuneOnlyClassifier, \
    VariantFiveFineTuneClassifier, VariantSixFineTuneClassifier, VariantSeventFineTuneOnlyClassifier, VariantEightFineTuneOnlyClassifier, \
    EnsembleModel
    


VARIANT_ONE_MODEL_PATH = 'model/patch_variant_1_finetune_1_epoch_best_model.sav'
VARIANT_TWO_MODEL_PATH = 'model/patch_variant_2_finetune_1_epoch_best_model.sav'
VARIANT_THREE_MODEL_PATH = 'model/patch_variant_3_finetune_1_epoch_best_model.sav'
VARIANT_FIVE_MODEL_PATH = 'model/patch_variant_5_finetune_1_epoch_best_model.sav'
VARIANT_SIX_MODEL_PATH = 'model/patch_variant_6_finetune_1_epoch_best_model.sav'
VARIANT_SEVEN_MODEL_PATH = 'model/patch_variant_7_finetune_1_epoch_best_model.sav'
VARIANT_EIGHT_MODEL_PATH = 'model/patch_variant_8_finetune_1_epoch_best_model.sav'
ENSEMBLE_MODEL_PATH = 'model/patch_ensemble_model.sav'


use_cuda = cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
random_seed = 109
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

def load_codebert_1():
    FINE_TUNED_MODEL_PATH = 'model/patch_variant_1_finetuned_model.sav'


    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    model = VariantOneFinetuneClassifier()

    if torch.cuda.device_count() > 1:
        # print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.load_state_dict(torch.load(FINE_TUNED_MODEL_PATH))
    code_bert = model.module.code_bert

    if torch.cuda.device_count() > 1:
        # print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        code_bert = nn.DataParallel(code_bert)
    code_bert = code_bert.to(device)

    code_bert.eval()


    return tokenizer, code_bert


def get_embedding_1(commit):

    print("Loading codebert for variant 1")

    tokenizer, codebert = load_codebert_1()

    diff = ''
    for file_code_change in commit:
        diff = diff + file_code_change  + '\n'

    removed_code = preprocess_finetuned_variant_1.get_code_version(diff, False)
    added_code = preprocess_finetuned_variant_1.get_code_version(diff, True)

    code = removed_code + tokenizer.sep_token + added_code

    embedding = preprocess_finetuned_variant_1.get_commit_embeddings([code], tokenizer, codebert)

    embedding = torch.FloatTensor(embedding)

    return embedding


def get_feature_1(embed_1):
    print("Extracting feature 1")
    
    model = VariantOneClassifier()
    if torch.cuda.device_count() > 1:
        # print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.load_state_dict(torch.load(VARIANT_ONE_MODEL_PATH))
    model.to(device)

    model.eval()

    embed_1 = embed_1.to(device)

    with torch.no_grad():
        outs = model(embedding_batch = embed_1, need_final_feature=True)
        features = outs[1]
        outs = outs[0]
        outs = F.softmax(outs, dim=1)
        prob = outs[0]
        return features
    
def load_codebert_2():

    FINE_TUNED_MODEL_PATH = 'model/patch_variant_2_finetuned_model.sav'

    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    model = VariantTwoFineTuneClassifier()
    if torch.cuda.device_count() > 1:
        # print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.load_state_dict(torch.load(FINE_TUNED_MODEL_PATH))
    code_bert = model.module.code_bert

    if torch.cuda.device_count() > 1:
        # print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        code_bert = nn.DataParallel(code_bert)
    code_bert = code_bert.to(device)

    code_bert.eval()


    return tokenizer, code_bert


def get_embedding_2(commit):

    print("Loading codebert for variant 2")

    tokenizer, codebert = load_codebert_2()

    code_list = []
    for diff in commit:
        removed_code = preprocess_finetuned_variant_2.get_code_version(diff, False)
        added_code = preprocess_finetuned_variant_2.get_code_version(diff, True)
        code = removed_code + tokenizer.sep_token + added_code
        code_list.append(code)

    embedding = preprocess_finetuned_variant_2.get_file_embeddings(code_list, tokenizer, codebert)


    # truncate and padding embedding 

    empty_code = tokenizer.sep_token + ''
    inputs = tokenizer([empty_code], padding=True, max_length=512, truncation=True, return_tensors="pt")
    input_ids, attention_mask = inputs.data['input_ids'], inputs.data['attention_mask']
    empty_embedding = codebert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[0, 0, :].tolist()

    max_data_length = 5
    if len(embedding) > max_data_length:
        embedding = embedding[:max_data_length]
    while len(embedding) < max_data_length:
        embedding.append(empty_embedding)

    embedding = torch.FloatTensor([embedding])

    return embedding


def get_feature_2(embed_2):
    print("Extracting feature 2")
    model = VariantTwoClassifier()

    if torch.cuda.device_count() > 1:
        # print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)

    model.load_state_dict(torch.load(VARIANT_TWO_MODEL_PATH))

    model.eval()

    embed_2 = embed_2.to(device)
    
    with torch.no_grad():
        outs = model(file_batch=embed_2, need_final_feature=True)
        features = outs[1]
        outs = outs[0]

        outs = F.softmax(outs, dim=1)
        prob = outs[0]
        return features


def load_codebert_3():

    FINE_TUNED_MODEL_PATH = 'model/patch_variant_3_finetuned_model.sav'
    
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

    model = VariantThreeFineTuneOnlyClassifier()
    if torch.cuda.device_count() > 1:
        # print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.load_state_dict(torch.load(FINE_TUNED_MODEL_PATH))
    code_bert = model.module.code_bert

    if torch.cuda.device_count() > 1:
        # print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        code_bert = nn.DataParallel(code_bert)
    code_bert = code_bert.to(device)

    code_bert.eval()

    return tokenizer, code_bert


def get_embedding_3(commit):

    print("Loading codebert for variant 3")

    tokenizer, codebert = load_codebert_3()

    hunk_list = []
    for diff in commit:
        hunk_list.extend(preprocess_finetuned_variant_3.get_hunk_from_diff(diff))

    code_list = []
    for hunk in hunk_list:
        removed_code = preprocess_finetuned_variant_3.get_code_version(hunk, False)
        added_code = preprocess_finetuned_variant_3.get_code_version(hunk, True)

        code = removed_code + tokenizer.sep_token + added_code
        
        code_list.append(code)

    embedding = preprocess_finetuned_variant_3.get_hunk_embeddings(code_list, tokenizer, codebert)
    embedding = torch.FloatTensor(embedding)

    # print(embedding.shape)

    if len(embedding) < 5:
        # print("padding")
        j, k = embedding.size(0), embedding.size(1)
        embedding = torch.cat(
            [embedding,
             torch.zeros((5 - j, k))])

    # print(embedding.shape)
    embedding = embedding.unsqueeze(0)
    
    # print(embedding.shape)

    return embedding


def get_feature_3(embed_3):

    print("Extracting feature 3")

    model = VariantThreeClassifier()
    if torch.cuda.device_count() > 1:
        # print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)

    model.load_state_dict(torch.load(VARIANT_THREE_MODEL_PATH))
    model.eval()

    with torch.no_grad():
        outs = model(code=embed_3, need_final_feature=True)
        features = outs[1]
        outs = outs[0]
        outs = F.softmax(outs, dim=1)
        prob = outs[0]
        return features
    

def load_codebert_5():

    FINE_TUNED_MODEL_PATH = 'model/patch_variant_5_finetuned_model.sav'

    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    model = VariantFiveFineTuneClassifier()
    if torch.cuda.device_count() > 1:
        # print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.load_state_dict(torch.load(FINE_TUNED_MODEL_PATH))
    code_bert = model.module.code_bert

    if torch.cuda.device_count() > 1:
        # print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        code_bert = nn.DataParallel(code_bert)
    code_bert = code_bert.to(device)

    code_bert.eval()

    return tokenizer, code_bert


def get_embedding_5(commit):

    print("Loading codebert for variant 5")

    tokenizer, codebert = load_codebert_5()
    
    diff = ''

    for file_change in commit:
        diff = diff + file_change + '\n'


    removed_code = tokenizer.sep_token + preprocess_finetuned_variant_5.get_code_version(diff, False)
    added_code = tokenizer.sep_token + preprocess_finetuned_variant_5.get_code_version(diff, True)

    removed_embedding = preprocess_finetuned_variant_5.get_commit_embeddings([removed_code], tokenizer, codebert)
    added_embedding = preprocess_finetuned_variant_5.get_commit_embeddings([added_code], tokenizer, codebert)

    removed_embedding = torch.FloatTensor(removed_embedding)
    added_embedding = torch.FloatTensor(added_embedding)

    return removed_embedding, added_embedding


def get_feature_5(removed_embedding, added_embedding):

    print("Extracting feature 5")
     
    model = VariantFiveClassifier()
    if torch.cuda.device_count() > 1:
        # print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)

    model.load_state_dict(torch.load(VARIANT_FIVE_MODEL_PATH))

    removed_embedding, added_embedding = removed_embedding.to(device), added_embedding.to(device)
    
    with torch.no_grad():
        outs = model(before_batch=removed_embedding, after_batch=added_embedding, need_final_feature=True)
        features = outs[1]
        outs = outs[0]
        outs = F.softmax(outs, dim=1)
        prob = outs[0]
        return features


def load_codebert_6():
    FINE_TUNED_MODEL_PATH = 'model/patch_variant_6_finetuned_model.sav'

    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    model = VariantSixFineTuneClassifier()

    if torch.cuda.device_count() > 1:
        # print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.load_state_dict(torch.load(FINE_TUNED_MODEL_PATH))
    code_bert = model.module.code_bert

    if torch.cuda.device_count() > 1:
        # print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        code_bert = nn.DataParallel(code_bert)
    code_bert = code_bert.to(device)

    code_bert.eval()

    return tokenizer, code_bert


def get_embedding_6(commit):

    print("Loading codebert for variant 6")

    tokenizer, codebert = load_codebert_6()

    removed_code_list = []
    added_code_list = []

    for diff in commit:
        removed_code = tokenizer.sep_token + preprocess_finetuned_variant_6.get_code_version(diff, False)
        added_code = tokenizer.sep_token + preprocess_finetuned_variant_6.get_code_version(diff, True)

        removed_code_list.append(removed_code)
        added_code_list.append(added_code)

    removed_embeddings = preprocess_finetuned_variant_6.get_file_embeddings(removed_code_list, tokenizer, codebert)
    added_embeddings = preprocess_finetuned_variant_6.get_file_embeddings(added_code_list, tokenizer, codebert)

    empty_code = tokenizer.sep_token + ''
    inputs = tokenizer([empty_code], padding=True, max_length=512, truncation=True, return_tensors="pt")
    input_ids, attention_mask = inputs.data['input_ids'], inputs.data['attention_mask']
    empty_embedding = codebert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[0, 0, :].tolist()

    if len(removed_embeddings) > 5:
            removed_embeddings = removed_embeddings[:5]
    if len(added_embeddings) > 5:
        added_embeddings = added_embeddings[:5]
    while len(removed_embeddings) < 5:
        removed_embeddings.append(empty_embedding)
    while len(added_embeddings) < 5:
        added_embeddings.append(empty_embedding)

    removed_embeddings = torch.FloatTensor([removed_embeddings])
    added_embeddings = torch.FloatTensor([added_embeddings])

    return removed_embeddings, added_embeddings


def get_feature_6(removed_embedding, added_embedding):

    print("Extracting feature 6")

    model = VariantSixClassifier()
    if torch.cuda.device_count() > 1:
        # print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)

    model.load_state_dict(torch.load(VARIANT_SIX_MODEL_PATH))

    model.eval()

    with torch.no_grad():

        removed_embedding, added_embedding = removed_embedding.to(device), added_embedding.to(device)
        outs = model(before_batch=removed_embedding, after_batch=added_embedding, need_final_feature=True)

        features = outs[1]
        outs = outs[0]
        
        outs = F.softmax(outs, dim=1)
        prob = outs[0]

        return features


def load_codebert_7():
    FINE_TUNED_MODEL_PATH = 'model/patch_variant_7_finetuned_model.sav'

    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    model = VariantSeventFineTuneOnlyClassifier()
    if torch.cuda.device_count() > 1:
        # print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.load_state_dict(torch.load(FINE_TUNED_MODEL_PATH))
    code_bert = model.module.code_bert

    if torch.cuda.device_count() > 1:
        # print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        code_bert = nn.DataParallel(code_bert)
    code_bert = code_bert.to(device)

    code_bert.eval()

    return tokenizer, code_bert


def get_embedding_7(commit):


    print("Loading codebert for variant 7")

    tokenizer, codebert = load_codebert_7()
    
    hunk_list =  []

    for diff in commit:
        hunk_list = preprocess_finetuned_variant_7.get_hunk_from_diff(diff)

    
    removed_code_list = []
    added_code_list = []

    
    for hunk in hunk_list:
        removed_code = preprocess_finetuned_variant_7.get_code_version(hunk, False)
        added_code = preprocess_finetuned_variant_7.get_code_version(hunk, True)

        removed_code_list.append(removed_code)
        added_code_list.append(added_code)

    if len(removed_code_list) == 0:
        removed_code_list = ['']

    if len(added_code_list) == 0:
        added_code_list = ['']

    removed_embeddings = preprocess_finetuned_variant_7.get_hunk_embeddings(removed_code_list, tokenizer, codebert)
    added_embeddings = preprocess_finetuned_variant_7.get_hunk_embeddings(added_code_list, tokenizer, codebert)

    removed_embeddings = torch.FloatTensor(removed_embeddings)
    added_embeddings = torch.FloatTensor(added_embeddings)

    if len(removed_embeddings) < 5:
        # print("padding")
        j, k = removed_embeddings.size(0), removed_embeddings.size(1)
        removed_embeddings = torch.cat(
            [removed_embeddings,
             torch.zeros((5 - j, k))])

    # print(embedding.shape)
    removed_embeddings = removed_embeddings.unsqueeze(0)

    if len(added_embeddings) < 5:
        # print("padding")
        j, k = added_embeddings.size(0), added_embeddings.size(1)
        added_embeddings = torch.cat(
            [added_embeddings,
             torch.zeros((5 - j, k))])

    # print(embedding.shape)
    added_embeddings = added_embeddings.unsqueeze(0)
        
    return removed_embeddings, added_embeddings


def get_feature_7(removed_embedding, added_embedding):

    print("Extracting feature 7")

    model = VariantSevenClassifier()
    if torch.cuda.device_count() > 1:
        # print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)

    model.load_state_dict(torch.load(VARIANT_SEVEN_MODEL_PATH))

    model.eval()

    with torch.no_grad():

        removed_embedding, added_embedding = removed_embedding.to(device), added_embedding.to(device)
        
        outs = model(before_batch=removed_embedding, after_batch=added_embedding, need_final_feature=True)
        features = outs[1]
        outs = outs[0]
        outs = F.softmax(outs, dim=1)
        prob = outs[0]

        return features


def load_codebert_8():

    FINE_TUNED_MODEL_PATH = 'model/patch_variant_8_finetuned_model.sav'

    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    model = VariantEightFineTuneOnlyClassifier()
    if torch.cuda.device_count() > 1:
        # print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.load_state_dict(torch.load(FINE_TUNED_MODEL_PATH))
    code_bert = model.module.code_bert

    if torch.cuda.device_count() > 1:
        # print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        code_bert = nn.DataParallel(code_bert)
    code_bert = code_bert.to(device)

    code_bert.eval()


    return tokenizer, code_bert



def get_embedding_8(commit):

    print("Loading codebert for variant 8")

    tokenizer, codebert = load_codebert_8()
    
    code = ''

    for diff in commit:
        code = code + diff + '\n'

    removed_code = preprocess_finetuned_variant_8.get_code_version(code, False)
    added_code = preprocess_finetuned_variant_8.get_code_version(code, True)

    removed_line_list = preprocess_finetuned_variant_8.get_line_from_code(tokenizer.sep_token, removed_code)
    added_line_list = preprocess_finetuned_variant_8.get_line_from_code(tokenizer.sep_token, added_code)

    if len(removed_line_list) == 0:
        removed_line_list = ['']

    if len(added_line_list) == 0:
        added_line_list = ['']

    removed_embeddings =preprocess_finetuned_variant_8.get_line_embeddings(removed_line_list, tokenizer, codebert)
    added_embeddings = preprocess_finetuned_variant_8.get_line_embeddings(added_line_list, tokenizer, codebert)


    removed_embeddings = torch.FloatTensor(removed_embeddings)

    added_embeddings = torch.FloatTensor(added_embeddings)

    if len(removed_embeddings) < 5:
        # print("padding")
        j, k = removed_embeddings.size(0), removed_embeddings.size(1)
        removed_embeddings = torch.cat(
            [removed_embeddings,
             torch.zeros((5 - j, k))])

    # print(embedding.shape)
    removed_embeddings = removed_embeddings.unsqueeze(0)

    if len(added_embeddings) < 5:
        # print("padding")
        j, k = added_embeddings.size(0), added_embeddings.size(1)
        added_embeddings = torch.cat(
            [added_embeddings,
             torch.zeros((5 - j, k))])

    # print(embedding.shape)
    added_embeddings = added_embeddings.unsqueeze(0)
        
    return removed_embeddings, added_embeddings


def get_feature_8(removed_embedding, added_embedding):

    print("Extracting feature 8")

    model = VariantEightClassifier()

    model.to(device)
    model.load_state_dict(torch.load(VARIANT_EIGHT_MODEL_PATH))
    model.eval()

    with torch.no_grad():
        removed_embedding, added_embedding = removed_embedding.to(device), added_embedding.to(device)
        outs = model(before_batch=removed_embedding, after_batch=added_embedding, need_final_feature=True)
        features = outs[1]
        outs = outs[0]
        outs = F.softmax(outs, dim=1)
        prob = outs[0]

        return features


def get_ensemble_prediction(feature_1, feature_2, feature_3, feature_5, feature_6, feature_7, feature_8):

    model = EnsembleModel(ablation_study=False, variant_to_drop=[])
    if torch.cuda.device_count() > 1:
            # print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            model = nn.DataParallel(model)
    model.load_state_dict(torch.load(ENSEMBLE_MODEL_PATH))
    model.to(device)

    feature_1 = feature_1.to(device)
    feature_2 = feature_2.to(device)
    feature_3 = feature_3.to(device)
    feature_5 = feature_5.to(device)
    feature_6 = feature_6.to(device)
    feature_7 = feature_7.to(device)
    feature_8 = feature_8.to(device)
    

    with torch.no_grad():
        model.eval()
        outs, pca_features = model(feature_1=feature_1, feature_2=feature_2, feature_3=feature_3, feature_5=feature_5, feature_6=feature_6, feature_7=feature_7, feature_8=feature_8, need_features=True)

        outs = F.softmax(outs, dim=1)
        prob = outs[:, 1].tolist()[0]

        return prob


def infer(commit, LOC):
    embed_1 = get_embedding_1(commit)
    feature_1 = get_feature_1(embed_1)

    embed_2 = get_embedding_2(commit)
    feature_2 = get_feature_2(embed_2)

    embed_3 = get_embedding_3(commit)
    feature_3 = get_feature_3(embed_3)

    removed_embedding, added_embedding = get_embedding_5(commit)
    feature_5 = get_feature_5(removed_embedding, added_embedding)
    
    removed_embedding, added_embedding = get_embedding_6(commit)
    feature_6 = get_feature_6(removed_embedding, added_embedding)

    removed_embedding, added_embedding = get_embedding_7(commit)
    feature_7 = get_feature_7(removed_embedding, added_embedding)

    removed_embedding, added_embedding = get_embedding_8(commit)
    feature_8 = get_feature_8(removed_embedding, added_embedding)

    prob = get_ensemble_prediction(feature_1, feature_2, feature_3, feature_5, feature_6, feature_7, feature_8)

    # print("Prob: before: {}".format(prob))
    prob = adjustment_runner.calculate_prob(prob, LOC)
    print("Predicted probability for being Vulnerability-fixing commit: {}".format(prob))

    return prob

# giang: just a sample commit
def get_sample_commit():
    commit = []

    # dataset_name = 'dataset_sample.csv'
    dataset_name = 'dataset_sample_2.csv'
    df = pd.read_csv(dataset_name)

    df = df[['commit_id', 'repo', 'partition', 'diff', 'label']]
    items = df.to_numpy().tolist()

    url_to_diff = {}

    print(items[0][0])
    for item in items:
        commit_id = item[0]
        # print(commit_id)
        repo = item[1]
        url = repo + '/commit/' + commit_id
        diff = item[3]
        commit.append(diff)

    # just a random number 

    LOC = 20

    return commit, LOC


# # test
# commit, LOC = get_sample_commit()

# infer(commit, LOC)