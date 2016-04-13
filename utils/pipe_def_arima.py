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

pipe_list_yield_base = ['FeatureSelector']
pipe_list_yield = pipe_list_yield_base + \
    ['MissingValueInputer', 'FillRemainingNaN']


############################
# kwargs_list_brick
kwargs_micro = base_kwargs.copy()
kwargs_micro.update({'FeatureSelector__features': data['micro_feats'].keys()})

kwargs_yield_base = {'FeatureSelector__features': ['yield']}


kwargs_yield = base_kwargs.copy()
kwargs_yield.update({'FeatureSelector__features': ['yield']})

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
# building the pipes for the output
pipe_list = {'yield': pipe_list_yield_base}
pipe_kwargs = {'yield': kwargs_yield_base}
pipe_yield_base = pipe(pipe_list, pipe_kwargs)

pipe_list = {'yield': pipe_list_yield}
pipe_kwargs = {'yield': kwargs_yield}
pipe_yield = pipe(pipe_list, pipe_kwargs)


############################
# building the pipes for features

############################
# pipe_micro_auto_input_arima
pipe_list_maia = ['FeatureSelector',
                  'AutoArimaInputer',
                  'MyStandardScaler']
kwargs_maia = {'FeatureSelector__features': data['micro_feats'].keys()}
pipe_list = {'maia': pipe_list_maia}
pipe_kwargs = {'maia': kwargs_maia}
pipe_maia = pipe(pipe_list, pipe_kwargs)

############################
# pipe_maia improved
pipe_list1 = ['FeatureSelector',
              'RemoveZeroValues',
              'AutoArimaInputer',
              'MyStandardScaler']
kwargs_list1 = {'FeatureSelector__features': ['humidity', 'temp']}

pipe_list2 = ['FeatureSelector',
              'AutoArimaInputer',
              'MyStandardScaler']
list_feats = [f for f in data['micro_feats'].keys() if f not in [
    'humidity', 'temp', 'leafwet460_min']]
kwargs_list2 = {'FeatureSelector__features': list_feats}

pipe_list = {'part1': pipe_list1,
             'part2': pipe_list2}
pipe_kwargs = {'part1': kwargs_list1,
               'part2': kwargs_list2}
pipe_maia_V2 = pipe(pipe_list, pipe_kwargs)


############################
# pipe0

# THE YIELD HAS ALWAYS ATO BE THE LAST ONE !!
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
               'macro_guel': kwargs_macro_guel}
pipe1 = pipe(pipe_list, pipe_kwargs)

############################
# pipe1_pca


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
