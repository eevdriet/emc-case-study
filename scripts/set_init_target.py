import json

import numpy as np
import pandas as pd

from emc.data.constants import *
from emc.log import setup_logger
from emc.util import Paths

logger = setup_logger(__name__)


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

        # Load monitor age data
        path = Paths.worm_data(worm, 'monitor_age', use_merged=True)
        n_age_cats = 1 if 'merged' in str(path) else N_AGE_CATEGORIES
        assert path.exists(), "Make sure to run the `merge` scripts"
        df_ma = pd.read_csv(path)

        # Load drug efficacy data and group per simulation
        path = Paths.worm_data(worm, 'drug_efficacy')
        drug_efficacy = pd.read_csv(path)
        drug_efficacy.reset_index(drop=True, inplace=True)
        drug_efficacy = drug_efficacy.groupby(['scenario', 'simulation'])

        # Load metadata
        path = Paths.worm_data(worm, 'meta_data')
        with open(path, 'r') as file:
            metadata = json.load(file)

        logger.info("Setting the expected infection level for scenarios...")
        for scenario in range(N_SCENARIOS):
            logger.info(f"Scenario {scenario + 1}")

            # Determine how often PC is applied
            data = metadata[scenario]
            mda_freq = data['mda_freq']

            for simulation in range(N_SIMULATIONS):
                logger.debug(f"\t {simulation + 1}/{N_SIMULATIONS}")
                # Determine which rows to take in the monitor age survey and get the corresponding drug efficacy survey
                start_ma = N_YEARS * n_age_cats * (N_SIMULATIONS * scenario + simulation)
                df_de = drug_efficacy.get_group((scenario + 1, simulation + 1)).reset_index(drop=True)

                for time in range(N_YEARS):
                    # Determine which rows to take in the drug efficacy survey
                    start_de = mda_freq * time
                    end_de = start_de + mda_freq - 1

                    # Determine the ERR column and take the first value if any
                    err = df_de.loc[start_de:end_de, 'ERR']
                    err_val = err.iloc[0] if err.first_valid_index() is not None else np.nan

                    # Add the ERR feature to the monitor age survey for each age category
                    for age_cat in range(n_age_cats):
                        df_ma.loc[start_ma + n_age_cats * time + age_cat, 'ERR'] = err_val

        # Save monitor age data with ERR feature
        df_ma.to_csv(path, index=False)


if __name__ == '__main__':
    set_target()
