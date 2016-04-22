import os
from tqdm import *
import pandas as pd
from helper import parse_conf_file


def get_score_model(path_model):
    try:
        score = [f for f in os.listdir(path_model) if f.split('_')[
            0] == 'train'][0].split('_')
        train, val = score[1], score[3]
    except:
        train, val = 100, 100
    return map(float, [train, val])


def get_result(root, hp=['h', 'reg', 'seq_length']):
    models = [f for f in os.listdir(root) if f != 'conf_model.json']
    results = []
    for model in tqdm(models, ncols=len(models)):
        path_model = os.path.join(root, model)
        try:
            conf = parse_conf_file(os.path.join(path_model, 'conf_model.json'))
            results.append([root, model] + [conf[key]
                                            for key in hp] + get_score_model(path_model))
        except:
            pass

    results = pd.DataFrame(
        results, columns=['root', 'model'] + hp + ['train', 'val'])
    results = results.sort_values(by='val')
    results.index = range(len(results))
    return results


def get_result_arima(root, hp=['h', 'reg', 'seq_length']):
    models = [f for f in os.listdir(root) if f != 'conf_model.json']
    results = []
    for model in tqdm(models, ncols=len(models)):
        path_model = os.path.join(root, model)
        try:
            conf = parse_conf_file(os.path.join(path_model, 'conf_model.json'))
            result_keys = ['rmse', 'aic', 'bic', 'hqic']
            score = {}
            score.update({'CV_train_%s' % (key): conf[
                         'CV_train_%s' % (key)] for key in result_keys})
            score.update({'CV_test_%s' % (key): conf[
                         'CV_test_%s' % (key)] for key in result_keys})
            results.append([root, model] +
                           [conf[key] for key in hp] +
                           score.values())
        except:
            pass

    results = pd.DataFrame(
        results, columns=['root', 'model'] + hp + score.keys())
    results = results.sort_values(by='CV_test_rmse')
    results.index = range(len(results))
    return results
