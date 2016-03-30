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
