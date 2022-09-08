from metrices_calculator import *
import argparse

import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)



def do_rq1():
    url_to_label, url_to_loc_mod = get_data()

    model_new_prob_java_path = 'probs/new_prob_java.txt'
    model_new_prob_python_path = 'probs/new_prob_python.txt'

    print('-' * 64)
    print('EVALUATING JAVA DATASET')
    calculate_auc(model_new_prob_java_path, url_to_label)
    calculate_effort(model_new_prob_java_path, 'java')
    calculate_normalized_effort(model_new_prob_java_path, 'java')
    
    print('-' * 64)
    print('-' * 64)
    print('EVALUATING PYTHON DATASET')

    calculate_auc(model_new_prob_python_path, url_to_label)
    calculate_effort(model_new_prob_python_path, 'python')
    calculate_normalized_effort(model_new_prob_python_path, 'python')

def do_rq2():

    url_to_label, url_to_loc_mod = get_data()

    model_prob_path_java = 'probs/prob_ensemble_classifier_test_java.txt'
    model_prob_path_python = 'probs/prob_ensemble_classifier_test_python.txt'
    model_new_prob_java_path = 'probs/new_prob_java.txt'
    model_new_prob_python_path = 'probs/new_prob_python.txt'


    print('-' * 64)
    print('EVALUATING MiDas NO ADJUSTMENT ON JAVA DATASET')
    calculate_auc(model_prob_path_java, url_to_label)
    calculate_effort(model_prob_path_java, 'java')
    calculate_normalized_effort(model_prob_path_java, 'java')
    
    print('-' * 64)
    print('-' * 64)
    print('EVALUATING MiDas NO ADJUSTMENT ON PYTHON DATASET')

    calculate_auc(model_prob_path_python, url_to_label)
    calculate_effort(model_prob_path_python, 'python')
    calculate_normalized_effort(model_prob_path_python, 'python')

    print('-' * 64)
    print('-' * 64)
    print('-' * 64)
    print('-' * 64)

    print('-' * 64)
    print('EVALUATING MiDas ON JAVA DATASET')
    calculate_auc(model_new_prob_java_path, url_to_label)
    calculate_effort(model_new_prob_java_path, 'java')
    calculate_normalized_effort(model_new_prob_java_path, 'java')
    
    print('-' * 64)
    print('-' * 64)
    print('EVALUATING MiDas ON PYTHON DATASET')

    calculate_auc(model_new_prob_python_path, url_to_label)
    calculate_effort(model_new_prob_python_path, 'python')
    calculate_normalized_effort(model_new_prob_python_path, 'python')


def do_rq3():
    pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Ensemble Classifier')
    parser.add_argument('--rq',
                        type=int,
                        default=1,
                        help='research question number, from 1')


    args = parser.parse_args()
    rq = args.rq
    if rq == 1:
        do_rq1()
    elif rq == 2:
        do_rq2()
    else:
        raise Exception("Invalid RQ number")

    