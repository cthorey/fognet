import os
import json
from time import strftime


class myDict(dict):
    ''' A special class to handle stuff '''

    def update(self, *args):
        dict.update(self, *args)
        return self


def dump_conf_file(config, fname):
    conf_file = os.path.join(fname, 'conf_model.json')
    with open(conf_file, 'w+') as f:
        json.dump(config, f,
                  sort_keys=True,
                  indent=4,
                  ensure_ascii=False)


def parse_conf_file(conf_file):
    ''' parse the configuration file'''

    with open(conf_file, 'r') as f:
        conf = json.load(f)
    return conf


def get_current_datetime():
    return strftime('%Y%m%d_%H%M%S')


def props(cls):
    ''' Return a dict of the attribut of the class without special method'''
    return {key: val for key, val in cls.__dict__.iteritems() if key[:1] != '_'}


def control_type_parameter(parameters):
    ''' return the good type for the parameters'''
    dict_type = {}
    dict_type.update(
        {f: int for f in ['nb_layers', 'stride', 'hiddens', 'seq_length']})
    dict_type.update({f: float for f in ['lr', 'reg']})
    dict_type.update({f: str for f in ['update_rule']})

    return {key: dict_type[key](val) for key, val in parameters.iteritems()}
