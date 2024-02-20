import pandas as pd
import numpy as np
import json

from emc.util import Paths
from emc.data.constants import *


def set_target() -> None:
    """
    Computes the initial target level for all simulations
    :return: Nothing, just updates the given monitor age CSV with an `target` column
    This column is based on the ERR in the corresponding drug efficacy survey
    Note that if the PC strategy is 2/year, the first ERR is chosen for that year
    E.g. for year 13 in the epidemiological survey, the ERR from time point 13.0 is taken
    """

    for worm in Worm:
        worm = worm.value

        path = Paths.worm_data(worm, 'monitor_age', use_merged=True)
        n_age_cats = 1 if 'merged' in str(path) else N_AGE_CATEGORIES
        assert path.exists(), "Make sure to run the `merge` scripts"
        df = pd.read_csv(path)

        path = Paths.worm_data(worm, 'drug_efficacy')
        df2 = pd.read_csv(path)
        df2.reset_index(drop=True, inplace=True)
        df2 = df2.groupby(['scenario', 'simulation'])

        path = Paths.worm_data(worm, 'meta_data')
        with open(path, 'r') as file:
            metadata = json.load(file)

        print("Setting the expected infection level for scenarios...")
        for scenario in range(N_SCENARIOS):
            print(f"\t- {scenario}")

            # Get right levels
            data = metadata[scenario]
            mda_freq = data['mda_freq']

            for sim in range(N_SIMULATIONS):
                start_ma = N_YEARS * n_age_cats * (N_SIMULATIONS * scenario + sim)
                df3 = df2.get_group((scenario + 1, sim + 1)).reset_index(drop=True)

                for time in range(N_YEARS):
                    start_de = mda_freq * time
                    series = df3.loc[start_de:start_de + mda_freq - 1, 'ERR']
                    err = series.iloc[0] if series.first_valid_index() is not None else np.nan

                    for age_cat in range(n_age_cats):
                        df.loc[start_ma + n_age_cats * time + age_cat, 'ERR'] = err

        df.to_csv(path, index=False)


if __name__ == '__main__':
    set_target()
