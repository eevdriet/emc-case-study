import logging

import pandas as pd
import numpy as np

from emc.data.constants import *
from emc.util import Paths
from emc.log import setup_logger

logger = setup_logger(__name__)


def weighted_mean(values: pd.Series, weights: pd.Series):
    """
    Calculate the mean of values based on their given weights
    :param values: Values to calculate mean for
    :param weights: Weights of the values
    :return: Weighted mean
    """
    if weights.sum() == 0:
        logger.warning("No weights defined, returning NaN")
        return np.nan

    return (weights * values).sum() / weights.sum()


def merge_age_cats() -> None:
    """
    Combines the CSV files generated from the `load_*.R` into one CSV
    :return: Nothing, just combines separate CSV files into one large one
    """

    # Parameters
    for worm in Worm:
        worm = worm.value

        # Load data
        path = Paths.worm_data(worm, 'monitor_age', use_merged=False)
        assert path.exists(), "Make sure to run the `merge_csv` script"
        df = pd.read_csv(path)

        # Go through all individual monitor_age dataframes
        logger.info("Merging the age categories for simulations per scenario...")

        for scenario in range(N_SCENARIOS):
            logger.info(f"Scenario {scenario + 1}")
            rows = []

            for simulation in range(N_SIMULATIONS):
                logger.debug(f"\t {simulation + 1}/{N_SIMULATIONS}")

                # Get index in data frame to start getting the rows from
                start_idx = N_YEARS * N_AGE_CATEGORIES * (N_SIMULATIONS * scenario + simulation)
                end_idx = start_idx + N_AGE_CATEGORIES - 1

                # Get the column values for all age categories per time point
                n_host = df.loc[start_idx:end_idx, 'n_host']
                n_host_eggpos = df.loc[start_idx:end_idx, 'n_host_eggpos']
                a_epg_obs = df.loc[start_idx:end_idx, 'a_epg_obs']
                a_drug_efficacy_true = df.loc[start_idx:end_idx, 'a_drug_efficacy_true']

                for time in range(N_YEARS):
                    # Aggregate the age categories rows into a single row based on the time point
                    col_values = {
                        'scenario': [scenario + 1],
                        'simulation': [simulation + 1],
                        'time': [time],
                        'n_host': [n_host.sum()],
                        'n_host_eggpos': [n_host_eggpos.sum()],
                        'a_epg_obs': [weighted_mean(a_epg_obs, n_host_eggpos)],
                        'a_drug_efficacy_true': [weighted_mean(a_drug_efficacy_true, n_host)]
                    }
                    assert all(col in df.columns for col in col_values.keys()), "Not all columns are copied!"

                    row = pd.DataFrame()
                    rows.append(row)
                    start_idx += N_AGE_CATEGORIES

            # Export per scenario (to speed up merging)
            df_merged = pd.concat(rows)
            path = Paths.data('csv') / f'{worm}_monitor_age_merged_{scenario}.csv'
            df_merged.to_csv(path, index=False)

        # Merge dataframes of all scenarios together
        df_merged = pd.DataFrame()

        logger.info("Merging the CSVs for scenarios...")
        for scenario in range(N_SCENARIOS):
            path = Paths.data('csv') / f'{worm}_monitor_age_merged_{scenario}.csv'
            df = pd.read_csv(path)

            # Merge and delete temporary scenario .csv file
            df_merged = pd.concat([df_merged, df])
            path.unlink()

        # Write the merged csv
        logger.info("Write merged CSV...")
        path = Paths.worm_data(worm, 'monitor_age', use_merged=True)
        df_merged.to_csv(path, index=False)


if __name__ == '__main__':
    merge_age_cats()
