import pandas as pd

from emc.data.constants import *
from emc.util import Paths


def add_features_group(group):
    """
    Add features per grouped data
    :param group: Group of data per simulation
    :return: Group with features added
    """
    group['inf_level'] = group['n_host_eggpos'] / group['n_host']

    return group


def add_features() -> None:
    """
    Add additional features to the CSV data files
    :return: Nothing, just adds to existing CSV files
    """

    # Load data
    for worm in Worm:
        worm = worm.value

        # Load data
        path = Paths.worm_data(worm, 'monitor_age', use_merged=True)
        assert path.exists(), "Make sure to run the `merge` scripts"
        df = pd.read_csv(path)

        # Add features per simulation
        df = df.groupby(['scenario', 'simulation']).apply(add_features_group).reset_index(drop=True)

        # Save data
        df.to_csv(path, index=False)


if __name__ == '__main__':
    # add_features()

    worm = Worm.ASCARIS.value
    path = Paths.worm_data(worm, 'drug_efficacy')
    df = pd.read_csv(path)
    df.rename(columns={'cost': 'cost_avg', 'costs': 'cost'}, inplace=True)
    df.to_csv(path)
