import logging

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
                path = Paths.host_data(worm, scenario, simulation)
                assert path.exists(), "Make sure to run the `load_drug_efficacy.R` script"

                df = pd.read_csv(path).reset_index(drop=True)
                dfs.append(df)

            df_combined = pd.concat(dfs)

            # Export per scenario (to speed up merging)
            path = Paths.data('csv') / f'{worm}_drug_efficacy_{scenario}.feather'
            df_combined.to_feather(path, index=False)

        # Merge dataframes of all scenarios together
        df_combined = pd.DataFrame()

        logger.info("Merging the CSVs for scenarios...")
        for scenario in range(N_SCENARIOS):
            path = Paths.data('csv') / f'{worm}_drug_efficacy_{scenario}.feather'
            df = pd.read_feather(path)

            # Merge and delete temporary scenario .csv file
            df_combined = pd.concat([df_combined, df])
            path.unlink()

        # Clean df by properly setting the index, fixing the col order and setting int cols as int
        df_combined.reset_index(drop=True, inplace=True)

        # Write the merged csv
        logger.info("Write merged CSV...")
        path = Paths.worm_data(worm, 'drug_efficacy')
        df_combined.to_feather(path.with_suffix('.feather'), index=False)


if __name__ == '__main__':
    merge_csv()
