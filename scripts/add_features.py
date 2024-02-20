import pandas as pd
import numpy as np

from emc.data.constants import *
from emc.util import Paths


def rate_of_change(col: pd.Series):
    new = col
    old = col.shift(1)

    if old == 0 or new - old == 0:
        return np.nan
    else:
        return (new - old) / old


def add_features_group(group):
    group['inf_level'] = group['n_host_eggpos'] / group['n_host']
    group['inf_level_change'] = group['inf_level'].pct_change()
    group['a_epg_obs_change'] = group['a_epg_obs'].pct_change()

    return group


def add_features() -> None:
    """
    Combines the CSV files generated from the `load_*.R` into one CSV
    :return: Nothing, just combines separate CSV files into one large one
    """

    # Load data
    for worm in Worm:
        worm = worm.value

        path = Paths.worm_data(worm, 'monitor_age', use_merged=True)
        assert path.exists(), "Make sure to run the `merge` scripts"

        df = pd.read_csv(path)

        # Add features

        df = df.groupby(['scenario', 'simulation']).apply(add_features_group).reset_index(drop=True)

        # Write back
        df.to_csv(path, index=False)


if __name__ == '__main__':
    add_features()
