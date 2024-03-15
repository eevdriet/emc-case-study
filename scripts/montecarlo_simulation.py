import pandas as pd
import numpy as np
from scipy.stats import gamma
from scipy.stats import poisson
import sys as sys

from emc.data.constants import *
from emc.util import Paths
from emc.log import setup_logger

logger = setup_logger(__name__)


# Works well: Shape 100000 and w = 1/2
class MCSimulation:
    # Parameters
    __SHAPE_SLIDE = sys.maxsize
    __N_SIMULATIONS = 5000

    def __init__(self, simulation: pd.DataFrame, worm: Worm):
        self.sim = simulation

        if worm == Worm.ASCARIS:
            self.__SHAPE_DAY = 0.510
        else:
            self.__SHAPE_DAY = 1.000

    def total_simulation(self):
        result = self.sim.copy()
        logger.info(f"Number of rows in DF: {len(result)}")

        for idx in range(len(result)):
            logger.info(f"i: {idx}")

            if result.at[idx, 'pre'] != 0 and not np.isnan(result.at[idx, 'pre']):
                result.at[idx, 'sim'] = self.__simulate_count_ids(result.at[idx, 'pre'])
            else:
                result.at[idx, 'pre_sim'] = 0

            if result.at[idx, 'post'] != 0 and not np.isnan(result.at[idx, 'post']):
                result.at[idx, 'post_sim'] = self.__simulate_count_ids(result.at[idx, 'post'])
            else:
                result.at[idx, 'post_sim'] = 0
        return result

    def __simulate_count_ids(self, mu_i: float):
        logger.info(f"\t- mu_i: {mu_i}")
        mu_id = gamma.rvs(self.__SHAPE_DAY, scale=mu_i / self.__SHAPE_DAY)
        logger.info(f"\t- mu_id: {mu_id}")
        mu_ids = gamma.rvs(self.__SHAPE_SLIDE, scale=mu_id / self.__SHAPE_SLIDE)
        logger.info(f"\t- mu_ids: {mu_ids}")
        count = poisson.rvs(mu_ids)
        logger.info(f"\t- count: {count}")

        return count

def main():
    # Import data
    worm = Worm.ASCARIS
    path = Paths.data() / 'result.feather'
    print(sys.maxsize)
    logger.info("Start Monte Carlo simulation")
    logger.info("Loading data...")
    df = pd.read_feather(path.with_suffix('.feather'))
    mc_simulation = MCSimulation(df, worm)

    df_result = mc_simulation.total_simulation()
    df_result.to_csv("result.csv")

    print(df_result)


if __name__ == '__main__':
    main()
