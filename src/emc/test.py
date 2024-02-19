import numpy as np
import json
from scipy import stats
from emc.data.constants import *
from emc.util import data_path
from itertools import product
from math import isnan

def show_statistics(levels):


    for baseline in levels:
        none_levels = levels[baseline]['none']
        for res_mode in levels[baseline]:
            if res_mode == 'none':
                continue
            comp_levels = levels[baseline][res_mode]
            for none_level, comp_level in zip(none_levels, comp_levels):
                none_mean, none_sd, *rest, none_n = none_level
                comp_mean, comp_sd, *rest, comp_n = comp_level

                if isnan(none_mean) or isnan(comp_mean):
                    continue
                
                t_stat, p_val = stats.ttest_ind_from_stats(none_mean, none_sd, none_n, comp_mean, comp_sd, comp_n)
                if p_val < 0.05:
                    hypothesis_rejected_count += 1
                else:
                    hypothesis_not_rejected_count += 1

    


def main():
    # data importeren uit de json files
    bucket_size = 10
    for worm in Worm:
        worm = worm.value
        mda_strategy = ['sac', 'community']
        mda_freq = [1, 2]

        for strat, freq in product(mda_strategy, mda_freq):
            freq_str = f'{freq}year'
            strat_str = strat
            fname = f'{worm}_{bucket_size}_{freq_str}_{strat_str}.json'

            path = data_path() / 'levels' / fname

            with open(path, 'r') as file:
                levels = json.load(file)

            show_statistics(levels)
    

if __name__ == "__main__":
    main()