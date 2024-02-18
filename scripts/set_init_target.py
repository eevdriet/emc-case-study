import pandas as pd
import numpy as np
import json

from emc.util import data_path
from emc.data.constants import *


def main() -> None:
    """
    Computes the initial target level for all simulations
    :return: Nothing, just updates the given monitor age CSV with an `target` column
    This column is based on the ERR in the corresponding drug efficacy survey
    Note that if the PC strategy is 2/year, the first ERR is chosen for that year
    E.g. for year 13 in the epidemiological survey, the ERR from time point 13.0 is taken
    """

    worm = Worm.ASCARIS
    path = data_path() / f'{worm}_monitor_age_merged.csv'
    df = pd.read_csv(path)

    drug_efficacy = pd.read_csv(data_path() / f'{worm}_drug_efficacy.csv').reset_index(drop=True)
    df2 = drug_efficacy.groupby(['scenario', 'simulation'])

    with open(data_path() / f'{worm}_metadata.json', 'r') as file:
        metadata = json.load(file)

    n_age_cats = 1 if 'merged' in str(path) else N_AGE_CATEGORIES

    for scenario in range(N_SCENARIOS):
        # Get right levels
        data = metadata[scenario]
        mda_freq = data['mda_freq']

        for sim in range(N_SIMULATIONS):
            start_ma = N_YEARS * n_age_cats * (N_SIMULATIONS * scenario + sim)
            df3 = df2.get_group((scenario + 1, sim + 1)).reset_index(drop=True)

            print(scenario, sim)

            for time in range(N_YEARS):
                start_de = mda_freq * time
                series = df3.loc[start_de:start_de + mda_freq - 1, 'ERR']
                target = series.iloc[0] if series.first_valid_index() is not None else np.nan

                for age_cat in range(n_age_cats):
                    df.loc[start_ma + n_age_cats * time + age_cat, 'target'] = target

    df.to_csv(path, index=False)


if __name__ == '__main__':
    main()
