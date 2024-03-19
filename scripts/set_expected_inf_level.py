import json

import pandas as pd

from emc.data.constants import *
from emc.log import setup_logger
from emc.util import Paths

logger = setup_logger(__name__)


def set_expected_infection_level() -> None:
    """
    Computes the expected infected level for all simulations
    :return: Nothing, just updates the given CSV with an `exp_inf_level` column
    This column is based on
    - At a given time step (time), what is the expected infected level
    given the infection level at the PREVIOUS step (prev)
    - The expected infected level is based on the levels from the relevant scenario with 5% buckets
    """
    for worm in Worm:
        worm = worm.value

        # Load monitor age data
        path = Paths.worm_data(worm, 'monitor_age', use_merged=True)
        monitor_age = pd.read_csv(path)
        df = monitor_age.sort_values(['scenario', 'simulation', 'time']).reset_index(drop=True)
        assert "inf_level" in df.columns, "Make sure to run the `add_features` script"

        # Load metadata
        path = Paths.worm_data(worm, 'metadata')
        with open(path, 'r') as file:
            metadata = json.load(file)

        bucket_size = 5
        n_age_cats = 1 if 'merged' in str(path) else N_AGE_CATEGORIES

        logger.info("Setting the expected infection level for scenarios...")
        for scenario in range(N_SCENARIOS):
            logger.info(f"Scenario {scenario + 1}")

            # Get levels from the right JSON file
            data = metadata[scenario]
            mda_freq = data['mda_freq']
            mda_strategy = data['mda_strategy']

            path = Paths.levels(worm, bucket_size=bucket_size, mda_freq=mda_freq, mda_strategy=mda_strategy)
            assert path.exists(), "Make sure to run the `build_levels` script in LevelBuilder"

            with open(path, 'r') as file:
                levels = json.load(file)

            for sim in range(N_SIMULATIONS):
                logger.debug(f"\t {sim + 1}/{N_SIMULATIONS}")
                start = n_age_cats * N_YEARS * (N_SIMULATIONS * scenario + sim)

                for time in range(N_YEARS):
                    prev = time - 1 if time > 0 else 0

                    for age_cat in range(n_age_cats):
                        inf_level = df.loc[start + n_age_cats * prev + age_cat, 'inf_level']

                        best_baseline = "0"
                        best_offset = float('inf')

                        for baseline in range(0, 100, bucket_size):
                            if str(baseline) not in levels:
                                break

                            level = levels[str(baseline)]["none"][prev][0]
                            offset = abs(level - inf_level)
                            if offset < best_offset:
                                logger.debug(f"Offset {offset} < {best_offset} -> use baseline {baseline}")
                                best_baseline = baseline
                                best_offset = offset

                        best_level = levels[str(best_baseline)]["none"][time][0]
                        df.loc[start + n_age_cats * time + age_cat, 'exp_inf_level'] = best_level

        path = Paths.worm_data(worm, 'monitor_age', use_merged=True)
        df.to_csv(path, index=False)


if __name__ == '__main__':
    set_expected_infection_level()
