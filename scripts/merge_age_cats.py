import pandas as pd
import numpy as np
from pathlib import Path

from emc.data.constants import *


def main() -> None:
    """
    Combines the CSV files generated from the `load_*.R` into one CSV
    :return: Nothing, just combines separate CSV files into one large one
    """

    # Parameters
    worm = Worm.ASCARIS
    data_path = Path.cwd() / f'{worm}_monitor_age.csv'
    df = pd.read_csv(data_path)

    def weighted_mean(values: pd.Series, weights: pd.Series):
        if weights.sum() == 0:
            return np.nan

        return (weights * values).sum() / weights.sum()

    # Go through all individual monitor_age dataframes
    for scenario in range(N_SCENARIOS):
        df_merged = pd.DataFrame()

        for simulation in range(N_SIMULATIONS):
            start = N_YEARS * N_AGE_CATEGORIES * (N_SIMULATIONS * scenario + simulation)
            print(scenario, simulation)

            for time in range(N_YEARS):
                n_host = df.loc[start:start + N_AGE_CATEGORIES - 1, 'n_host']
                n_host_eggpos = df.loc[start:start + N_AGE_CATEGORIES - 1, 'n_host_eggpos']
                a_epg_obs = df.loc[start:start + N_AGE_CATEGORIES - 1, 'a_epg_obs']
                row = pd.DataFrame({
                    'scenario': [scenario + 1],
                    'simulation': [simulation + 1],
                    'time': [time],
                    'n_host': [n_host.sum()],
                    'n_host_eggpos': [n_host_eggpos.sum()],
                    'a_epg_obs': [weighted_mean(a_epg_obs, n_host_eggpos)]
                })

                df_merged = pd.concat([df_merged, row])
                start += N_AGE_CATEGORIES

        # Export per scenario (to speed up merging)
        path = Path.cwd() / f'{worm}_monitor_age_merged_{scenario}.csv'
        df_merged.to_csv(path, index=False)

    # Merge dataframes of all scenarios together
    df_merged = pd.DataFrame()

    for scenario in range(N_SCENARIOS):
        path = Path.cwd() / f'{worm}_monitor_age_merged_{scenario}.csv'
        df = pd.read_csv(path)

        # Merge and delete temporary scenario .csv file
        df_merged = pd.concat([df_merged, df])
        path.unlink()

    # Write the merged csv
    data_path = Path.cwd() / f'{worm}_monitor_age_merged.csv'
    df_merged.to_csv(data_path, index=False)


if __name__ == '__main__':
    main()
