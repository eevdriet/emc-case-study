import logging
from math import ceil, floor

import numpy as np
import pandas as pd

from emc.data.constants import *
from emc.log import setup_logger
from emc.util import Paths

logger = setup_logger(__name__)


def merge_csv() -> None:
    """
    Combines the CSV files generated from the `load_*.R` into one CSV
    :return: Nothing, just combines separate CSV files into one large one
    """
    assert Paths.data('csv').exists(), "Make sure to create a 'csv' directory under the 'data' directory"

    logger = logging.getLogger(__name__)

    # Columns and their data type that should be included into the merged CSV
    cols = ['scenario', 'simulation', 'time', 'age_cat', 'n_host', 'n_host_eggpos', 'a_epg_obs',
            'a_drug_efficacy_true']
    dtypes = {'scenario': int, 'simulation': int, 'time': int, 'age_cat': int, 'n_host': 'Int64',
              'n_host_eggpos': 'Int64'}

    for worm in Worm:
        worm = worm.value

        # Go through all individual monitor_age dataframes
        logger.info("Merging the CSVs for simulations per simulation...")
        for scenario in range(N_SCENARIOS):
            logger.info(f"Scenario {scenario + 1}")
            dfs = []

            for simulation in range(N_SIMULATIONS):
                logger.debug(f"\t- {simulation + 1}/{N_SIMULATIONS}")
                # Load CSV for a single simulation
                path = Paths.data('csv') / f'{worm}_monitor_ageSC{scenario + 1:02}SIM{simulation + 1:04}.csv'
                assert path.exists(), "Make sure to run the `load_monitor_age.R` script"
                df = pd.read_csv(path).reset_index(drop=True)

                # Verify whether additional rows need to be added for missing years
                last_year = df.loc[df.index[-1], 'time']
                if last_year != N_YEARS - 1:
                    # Make sure the last_year_c is the last valid set year
                    last_year_ceil = ceil(last_year)
                    last_valid_year = df.loc[df.index[-len(AGE_CATEGORIES) - 1], 'time']

                    if last_year != last_year_ceil and floor(last_year) != last_valid_year:
                        last_year_ceil -= 1

                    # Round the last year for the last observation of each age category
                    for idx in range(len(AGE_CATEGORIES)):
                        df.loc[df.index[-idx - 1], 'time'] = last_year_ceil

                    # Add additional rows
                    rows = []
                    for year in range(last_year_ceil + 1, N_YEARS):
                        for age_cat in AGE_CATEGORIES:
                            col_values = {'scenario': scenario + 1,
                                          'simulation': simulation + 1,
                                          'time': [year],
                                          'age_cat': [age_cat],
                                          'n_host': [np.nan],
                                          'n_host_eggpos': [np.nan],
                                          'a_epg_obs': [np.nan],
                                          'a_drug_efficacy_true': [np.nan]}
                            assert all(col in cols for col in col_values.keys()), "Not all columns are copied!"

                            row = pd.DataFrame(cols)
                            rows.append(row)

                    df = pd.concat([df] + rows)

                # Combine all rows and make sure each simulation has the correct number of simulated years
                assert len(df) == N_AGE_CATEGORIES * N_YEARS
                dfs.append(df)

            df_combined = pd.concat(dfs)

            # Export per scenario (to speed up merging)
            path = Paths.data('csv') / f'{worm}_monitor_age_{scenario}.csv'
            df_combined.to_csv(path, index=False)

        # Merge dataframes of all scenarios together
        df_combined = pd.DataFrame()

        logger.info("Merging the CSVs for scenarios...")
        for scenario in range(N_SCENARIOS):
            path = Paths.data('csv') / f'{worm}_monitor_age_{scenario}.csv'
            df = pd.read_csv(path)

            # Merge and delete temporary scenario .csv file
            df_combined = pd.concat([df_combined, df])
            path.unlink()

        # Clean df by properly setting the index, fixing the col order and setting int cols as int
        df_combined.reset_index(drop=True, inplace=True)

        df_combined = df_combined[cols]
        df_combined = df_combined.astype(dtypes)

        # Write the merged csv
        logger.info("Write merged CSV...")
        path = Paths.worm_data(worm, 'monitor_age', use_merged=False)
        df_combined.to_csv(path, index=False)


if __name__ == '__main__':
    merge_csv()
