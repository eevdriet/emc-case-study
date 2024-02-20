import pandas as pd
import json

from emc.util import worm_path, data_path
from emc.data.constants import *


def main() -> None:
    """
    Computes the expected infected level for all simulations
    :return: Nothing, just updates the given CSV with an `exp_inf_level` column
    This column is based on
    - At a given time step (time), what is the expected infected level
    given the infection level at the PREVIOUS step (prev)
    - The expected infected level is based on the levels from the relevant scenario with 5% buckets
    """

    for worm in Worm:
        worm = worm.value

        path = worm_path(worm, 'monitor_age', use_merged=True)
        monitor_age = pd.read_csv(path)
        df = monitor_age.sort_values(['scenario', 'simulation', 'time']).reset_index(drop=True)

        with open(worm_path(worm, 'metadata'), 'r') as file:
            metadata = json.load(file)

        bucket_size = 5
        n_age_cats = 1 if 'merged' in str(path) else N_AGE_CATEGORIES

        for scenario in range(N_SCENARIOS):
            # Get right levels
            data = metadata[scenario]
            mda_freq = data['mda_freq']
            mda_strategy = data['mda_strategy']

            freq_str = f'{mda_freq}year' if mda_freq else 'any_freq'
            strategy_str = mda_strategy if mda_strategy else 'anyone'
            file_name = f'{worm}_{bucket_size}_{freq_str}_{strategy_str}.json'

            with open(data_path() / 'levels' / file_name, 'r') as file:
                levels = json.load(file)

            for sim in range(N_SIMULATIONS):
                start = n_age_cats * N_YEARS * (N_SIMULATIONS * scenario + sim)

                print(scenario, sim)

                for time in range(N_YEARS):
                    prev = time - 1 if time > 0 else 0

                    for age_cat in range(n_age_cats):
                        inf_level = df.loc[start + n_age_cats * prev + age_cat, 'inf_level']

                        best_baseline = "0"
                        best_offset = float('inf')

                        for baseline in range(0, 100, bucket_size):
                            if str(baseline) not in levels:
                                break

                            level = levels[str(baseline)]["none"][prev][0]
                            offset = abs(level - inf_level)
                            if offset < best_offset:
                                best_baseline = baseline
                                best_offset = offset

                        best_level = levels[str(best_baseline)]["none"][time][0]
                        df.loc[start + n_age_cats * time + age_cat, 'exp_inf_level'] = best_level

        df.to_csv(path, index=False)


if __name__ == '__main__':
    main()
