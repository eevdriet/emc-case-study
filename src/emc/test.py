import numpy as np
import json
from scipy import stats
from emc.data.constants import *
from emc.util import data_path
from itertools import product

# data importeren uit de json files
bucket_size = 10
for worm in Worm:
    mda_strategy = ['sac', 'community']
    mda_freq = [1, 2]

    for strat, freq in product(mda_strategy, mda_freq):
        freq_str = f'{freq}year' if freq else 'any_freq'
        strat_str = strat if strat else 'anyone'
        fname = f'{worm}_{bucket_size}_{freq_str}_{strat_str}.json'

        path = data_path() / 'levels' / fname

        with open(path, 'r') as file:
            levels = json.load(file)


def show_statistics(levels):
    for baseline in levels:
        none_levels = levels[baseline]['none']
        for res_mode in levels[baseline]:
            if res_mode == 'none':
                continue
            comp_levels = levels[baseline][res_mode]
            for none_level, comp_level in zip(none_levels, comp_levels):
                
            

    # Welch test
    t_stat, p_val = stats.ttest_ind(data1, data2, equal_var=False)