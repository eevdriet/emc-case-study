import json
from scipy import stats
from emc.data.constants import *
from emc.util import Paths
from itertools import product
from collections import defaultdict
from math import isnan


def show_statistics(levels):
    """
    Shows the statistics for the levels of a given scenario
    Not rejected -> Infection level means are equal between 'none' and the other resistance mode
    Rejected -> Infection level means are different between 'none' and the other resistance mode
    :param levels: Infection levels
    :return: Statistics
    """
    # Keep track of which tests are (not) rejected
    n_rejected = 0
    n_not_rejected = 0

    res_not_rejected = defaultdict(int)
    res_rejected = defaultdict(int)

    for baseline in levels:
        none_levels = levels[baseline]['none']
        for res_mode in levels[baseline]:
            # Only compare other resistance modes to 'none'
            if res_mode == 'none':
                continue

            # Find levels to compare to
            comp_levels = levels[baseline][res_mode]
            for none_level, comp_level in zip(none_levels, comp_levels):
                none_mean, none_sd, *rest, none_n = none_level
                comp_mean, comp_sd, *rest, comp_n = comp_level

                # Skip missing data
                if isnan(none_mean) or isnan(comp_mean):
                    continue

                # Perform T-test to verify whether the means are equal between 'none' and the other resistance mode
                t_stat, p_val = stats.ttest_ind_from_stats(none_mean, none_sd, none_n, comp_mean, comp_sd, comp_n)

                # Update statistics
                if p_val < 0.05:
                    n_rejected += 1
                    res_rejected[res_mode] += 1
                else:
                    n_not_rejected += 1
                    res_not_rejected[res_mode] += 1

    return {'not_rejected': n_not_rejected, 'rejected': n_rejected}, res_rejected, res_not_rejected


def main():
    bucket_size = 10

    # Initialize the statistics
    stats = {}
    for res_mode in ['none', 'dominant', 'codominant', 'recessive']:
        stats[res_mode] = {'not_rejected': 0, 'rejected': 0}

    # Go through all scenarios
    for worm in Worm:
        worm = worm.value
        mda_strategy = ['sac', 'community']
        mda_freq = [1, 2]

        for strat, freq in product(mda_strategy, mda_freq):
            key = (worm, strat, freq)

            # Load the levels file
            freq_str = f'{freq}year'
            strat_str = strat
            fname = f'{worm}_{bucket_size}_{freq_str}_{strat_str}.json'
            path = Paths.data('levels') / fname

            with open(path, 'r') as file:
                levels = json.load(file)

            # Update the statistics for the given levels
            stat, res_rejected, res_not_rejected = show_statistics(levels)

            for res_mode, count in res_rejected.items():
                stats[res_mode]['rejected'] += count
            for res_mode, count in res_not_rejected.items():
                stats[res_mode]['not_rejected'] += count

            stats[str(key)] = stat

    path = Paths.data('statistics') / 'stats.json'
    with open(path, 'w') as file:
        json.dump(stats, file, allow_nan=True, indent=4)


if __name__ == "__main__":
    main()
