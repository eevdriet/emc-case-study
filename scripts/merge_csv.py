import pandas as pd
import numpy as np
from math import ceil, floor

from emc.data.constants import *
from emc.util import Paths


def merge_csv() -> None:
    """
    Combines the CSV files generated from the `load_*.R` into one CSV
    :return: Nothing, just combines separate CSV files into one large one
    """
    assert Paths.data('csv').exists(), "Make sure to create a 'csv' directory under the 'data' directory"

    for worm in Worm:
        worm = worm.value

        # Go through all individual monitor_age dataframes
        print("Merging the CSVs for scenarios...")
        for scenario in range(N_SCENARIOS):
            print(f"\t- {scenario}")
            dfs = []

            for simulation in range(N_SIMULATIONS):
                path = Paths.data('csv') / f'{worm}_monitor_ageSC{scenario + 1:02}SIM{simulation + 1:04}.csv'
                assert path.exists(), "Make sure to run the `load_monitor_age.R` script"

                df = pd.read_csv(path).reset_index(drop=True)

                # Verify whether additional rows need to be added for missing years
                last_year = df.loc[df.index[-1], 'time']
                if last_year != N_YEARS - 1:
                    # Make sure the last_year_c is the last valid set year
                    last_year_c = ceil(last_year)
                    if last_year != last_year_c and floor(last_year) != df.loc[
                        df.index[-len(AGE_CATEGORIES) - 1], 'time']:
                        last_year_c -= 1

                    # Round the last year for the last observations
                    for idx in range(len(AGE_CATEGORIES)):
                        df.loc[df.index[-idx - 1], 'time'] = last_year_c

                    # Add additional rows
                    rows = []
                    for year in range(last_year_c + 1, N_YEARS):
                        for age_cat in AGE_CATEGORIES:
                            row = pd.DataFrame(
                                {'age_cat': [age_cat], 'time': [year], 'n_host': [np.nan], 'n_host_eggpos': [np.nan],
                                 'a_epg_obs': [np.nan], 'scenario': scenario + 1, 'simulation': simulation + 1,
                                 'a_drug_efficacy_true_': [np.nan]})
                            rows.append(row)

                    df = pd.concat([df] + rows)

                assert len(df) == N_AGE_CATEGORIES * N_YEARS
                dfs.append(df)

            df_combined = pd.concat(dfs)

            # Export per scenario (to speed up merging)
            path = Paths.data('csv') / f'{worm}_monitor_age_{scenario}.csv'
            df_combined.to_csv(path, index=False)

        # Merge dataframes of all scenarios together
        df_combined = pd.DataFrame()

        for scenario in range(N_SCENARIOS):
            path = Paths.data('csv') / f'{worm}_monitor_age_{scenario}.csv'
            df = pd.read_csv(path)

            # Merge and delete temporary scenario .csv file
            df_combined = pd.concat([df_combined, df])
            path.unlink()

        # Clean df by properly setting the index, fixing the col order and setting int cols as int
        df_combined.reset_index(drop=True, inplace=True)
        cols = ['scenario', 'simulation', 'time', 'age_cat', 'n_host', 'n_host_eggpos', 'a_epg_obs',
                'a_drug_efficacy_true']
        dtypes = {'scenario': int, 'simulation': int, 'time': int, 'age_cat': int, 'n_host': 'Int64',
                  'n_host_eggpos': 'Int64'}

        df_combined = df_combined[cols]
        df_combined = df_combined.astype(dtypes)

        # Write the merged csv
        path = Paths.worm_data(worm, 'monitor_age', use_merged=False)
        df_combined.to_csv(path, index=False)


if __name__ == '__main__':
    merge_csv()
