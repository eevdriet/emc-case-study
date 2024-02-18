import pandas as pd
from pathlib import Path
import json


def main() -> None:
    """
    Computes the expected infected level for all simulations
    :return: Nothing, just updates the given CSV with an `exp_inf_level` column
    This column is based on
    - At a given time step (time), what is the expected infected level
    given the infection level at the PREVIOUS step (prev)
    - The expected infected level is based on the levels from the relevant scenario with 5% buckets
    """

    worm = 'ascaris'
    data_path = Path.cwd() / '..' / 'data'
    path = data_path / f'{worm}_monitor_age_merged.csv'
    monitor_age = pd.read_csv(path)
    df = monitor_age.sort_values(['scenario', 'simulation', 'time']).reset_index(drop=True)

    with open(data_path / f'{worm}_metadata.json', 'r') as file:
        metadata = json.load(file)

    bucket_size = 5
    n_age_cats = 1 if 'merged' in str(path) else 4
    n_years = 21

    for scen in range(16):
        # Get right levels
        data = metadata[scen]
        mda_freq = data['mda_freq']
        mda_strategy = data['mda_strategy']

        freq_str = f'{mda_freq}year' if mda_freq else 'any_freq'
        strat_str = mda_strategy if mda_strategy else 'anyone'
        fname = f'{worm}_{bucket_size}_{freq_str}_{strat_str}.json'

        with open(data_path / 'levels' / fname, 'r') as file:
            levels = json.load(file)

        for sim in range(1000):
            start = n_age_cats * n_years * (1000 * scen + sim)

            print(scen, sim)

            for time in range(n_years):
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
