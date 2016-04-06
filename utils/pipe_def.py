from data_utils import load_raw_data

# data
data = load_raw_data()

############################
# base
base_kwargs = {'MissingValueInputer__method': 'time',
               'FillRemainingNaN__method': 'bfill'}

############################
# pipe_list_brick
pipe_list_micro = ['FeatureSelector',
                   'MissingValueInputer',
                   'FillRemainingNaN',
                   'MyStandardScaler']

pipe_list_macro = ['FeatureSelector',
                   'NumericFeatureSelector',
                   'MissingValueInputer',
                   'FillRemainingNaN',
                   'MyStandardScaler']

############################
# kwargs_list_brick
kwargs_micro = base_kwargs.copy()
kwargs_micro.update({'FeatureSelector__features': data['micro_feats'].keys()})

kwargs_macro_aga = base_kwargs.copy()
kwargs_macro_aga .update(
    {'FeatureSelector__features': data['aga_feats'].keys()})

# A cause du resampling, pas toutes les features dans sidi
sidi_feats = ['T', 'Po', 'P', 'Pa', 'U', 'Ff',
              'Tn', 'Tx', 'VV', 'Td', 'tR', 'Tg', 'sss']
sidi_feats = ['sidi_%s' % (f) for f in sidi_feats]
kwargs_macro_sidi = base_kwargs.copy()
kwargs_macro_sidi.update(
    {'FeatureSelector__features': sidi_feats})

kwargs_macro_guel = base_kwargs.copy()
kwargs_macro_guel.update(
    {'FeatureSelector__features': data['guel_feats'].keys()})


# Class to ease the process
class Pipe(object):

    def __call__(self, pipe_list, pipe_kwargs):
        return {'pipe_list': pipe_list,
                'pipe_kwargs': pipe_kwargs}
pipe = Pipe()

############################
# building the pipes

############################
# pipe0
pipe_list = {'micro': pipe_list_micro}
pipe_kwargs = {'micro': kwargs_micro}
pipe0 = pipe(pipe_list, pipe_kwargs)

############################
# pipe1
pipe_list = {'micro': pipe_list_micro,
             'macro_aga': pipe_list_macro,
             'macro_sidi': pipe_list_macro,
             'macro_guel': pipe_list_macro}
pipe_kwargs = {'micro': kwargs_micro,
               'macro_aga': kwargs_macro_aga,
               'macro_sidi': kwargs_macro_sidi,
               'macro_guel': kwargs_macro_guel
               }
pipe1 = pipe(pipe_list, pipe_kwargs)

############################
# pipe2
pipe_list = {'micro': pipe_list_micro,
             'macro_aga': pipe_list_macro,
             'macro_guel': pipe_list_macro}
pipe_kwargs = {'micro': kwargs_micro,
               'macro_aga': kwargs_macro_aga,
               'macro_guel': kwargs_macro_guel
               }
pipe2 = pipe(pipe_list, pipe_kwargs)

############################
# pipe3
pipe_list3 = ['FeatureSelector',
              'MissingValueInputer',
              'FillRemainingNaN']

pipe_list = {'micro': pipe_list3}
pipe_kwargs = {'micro': kwargs_micro}
pipe3 = pipe(pipe_list, pipe_kwargs)
