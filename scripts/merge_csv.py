import pandas as pd
import numpy as np
from pathlib import Path
from math import ceil, floor

from emc.data.constants import *


def main() -> None:
    """
    Combines the CSV files generated from the `load_*.R` into one CSV
    :return: Nothing, just combines separate CSV files into one large one
    """

    # Parameters
    worm = Worm.ASCARIS
    csv_path = Path.cwd() / '..' / 'csv'

    # Go through all individual monitor_age dataframes
    for scenario in range(N_SCENARIOS):
        df_combined = pd.DataFrame()

        for simulation in range(N_SIMULATIONS):
            print(scenario, simulation)

            path = csv_path / f'{worm}_monitor_ageSC{scenario + 1:02}SIM{simulation + 1:04}.csv'
            df = pd.read_csv(path).reset_index(drop=True)

            # Verify whether additional rows need to be added for missing years
            last_year = df.loc[df.index[-1], 'time']
            if last_year != N_YEARS - 1:
                # Make sure the last_year_c is the last valid set year
                last_year_c = ceil(last_year)
                if last_year != last_year_c and floor(last_year) != df.loc[df.index[-len(AGE_CATEGORIES) - 1], 'time']:
                    last_year_c -= 1

                # Round the last year for the last observations
                for idx in range(len(AGE_CATEGORIES)):
                    df.loc[df.index[-idx - 1], 'time'] = last_year_c

                # Add additional rows
                for year in range(last_year_c + 1, N_YEARS):
                    for age_cat in AGE_CATEGORIES:
                        row = {'age_cat': [age_cat], 'time': [year], 'n_host': [np.nan], 'n_host_eggpos': [np.nan],
                               'a_epg_obs': [np.nan], 'scenario': scenario + 1, 'simulation': simulation + 1}
                        df = pd.concat([df, pd.DataFrame(row)])

            assert len(df) == N_AGE_CATEGORIES * N_YEARS
            df_combined = pd.concat([df_combined, df])

        # Export per scenario (to speed up merging)
        path = Path.cwd() / f'{worm}_monitor_age_{scenario}.csv'
        df_combined.to_csv(path, index=False)

    # Merge dataframes of all scenarios together
    df_combined = pd.DataFrame()

    for scenario in range(N_SCENARIOS):
        path = Path.cwd() / f'{worm}_monitor_age_{scenario}.csv'
        df = pd.read_csv(path)

        # Merge and delete temporary scenario .csv file
        df_combined = pd.concat([df_combined, df])
        path.unlink()

    # Clean df by properly setting the index, fixing the col order and setting int cols as int
    df_combined.reset_index(drop=True, inplace=True)
    cols = ['scenario', 'simulation', 'time', 'age_cat', 'n_host', 'n_host_eggpos', 'a_epg_obs']
    dtypes = {'scenario': int, 'simulation': int, 'time': int, 'age_cat': int, 'n_host': 'Int64',
              'n_host_eggpos': 'Int64'}

    df_combined = df_combined[cols]
    df_combined = df_combined.astype(dtypes)

    # Write the merged csv
    path = Path.cwd() / f'{worm}_monitor_age.csv'
    df_combined.to_csv(path, index=False)


if __name__ == '__main__':
    main()
