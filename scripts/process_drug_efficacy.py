import pandas as pd

from emc.data.constants import *
from emc.util import Paths


def process_data(worm: str) -> None:
    path = Paths.worm_data(worm, 'drug_efficacy').with_suffix('.feather')
    df = pd.read_feather(path)

    # Group by 'scenario', 'simulation' and 'time' and aggregate host data
    grouped = df.groupby(['scenario', 'simulation', 'time']).agg(
        true_a_pre=('pre', lambda x: x.mean(skipna=True)),
        true_a_post=('post', lambda x: x.mean(skipna=True)),
        true_total_pre=('pre', lambda x: x.sum(skipna=True)),
        true_total_post=('post', lambda x: x.sum(skipna=True)),
        total_useful_tests=('post', lambda x: len(x) - x.isna().sum()),
        skipped_NaN_tests=('post', lambda x: x.isna().sum())
    ).reset_index()

    # Calculate statistics
    grouped['ERR'] = (1 - (grouped['true_a_post'] / grouped['true_a_pre']))
    grouped['EPG_change'] = (grouped['true_a_post'] - grouped['true_a_pre']) / grouped['true_a_pre']

    # Write merged data frame as CSV
    grouped.to_csv(path, index=False)


if __name__ == '__main__':
    for worm in Worm:
        worm = worm.value

        process_data(worm)
