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

            if result.at[idx, 'true_a_pre'] != 0 and not np.isnan(result.at[idx, 'true_a_pre']):
                result.at[idx, 'true_a_pre_sim'] = self.__simulate_count_ids(result.at[idx, 'true_a_pre'])
            else:
                result.at[idx, 'true_a_pre_sim'] = 0

            if result.at[idx, 'true_a_post'] != 0 and not np.isnan(result.at[idx, 'true_a_post']):
                result.at[idx, 'true_a_post_sim'] = self.__simulate_count_ids(result.at[idx, 'true_a_post'])
            else:
                result.at[idx, 'true_a_post_sim'] = 0
        return result

    def __simulate_count_ids(self, mu_i: float):
        # TODO: Decide on using mu ids or not!!!
        mu_ids = self.__simulate_mu_ids(mu_i)  # mu_ids = self.__simulate_mu_ids(mu_i)
        count = 0

        for i in range(self.__N_SIMULATIONS):
            count += poisson.rvs(mu_ids)

        avg_count = count / self.__N_SIMULATIONS

        logger.info(f"count: {avg_count}")
        return avg_count

    def __simulate_mu_ids(self, mu_i: float):
        mu_id = self.__simulate_mu_id(mu_i)
        # mu_ids = 0
        # for i in range(self.__TOTAL_SIM):
        #     mu_ids += gamma.rvs(self.__SHAPE_SLIDE, scale = mu_id / self.__SHAPE_SLIDE) # self.__SHAPE_SLIDE / mu_id
        # print("mu_ids: ")
        # # print(mu_ids)
        # print(mu_ids / self.__TOTAL_SIM)
        # return mu_ids / self.__TOTAL_SIM
        mu_ids = gamma.rvs(self.__SHAPE_SLIDE, scale=mu_id / self.__SHAPE_SLIDE)
        logger.info(f"mu_ids: {mu_ids}")

        return mu_ids

    def __simulate_mu_id(self, mu_i: float):
        # print("mu: ")
        # print(mu_i)
        # mu_id = 0
        # for i in range(self.__TOTAL_SIM):
        #     mu_id += gamma.rvs(self.__SHAPE_DAY, scale = mu_i / self.__SHAPE_DAY )#  (mu_i / self.__SHAPE_DAY)*(1/24)
        # print("mu_id: ")
        # print(mu_id / self.__TOTAL_SIM)
        # return mu_id / self.__TOTAL_SIM

        logger.info(f"mu: {mu_i}")
        return gamma.rvs(self.__SHAPE_DAY, scale=mu_i / self.__SHAPE_DAY)


def main():
    # Import data
    worm = Worm.ASCARIS
    path = Paths.worm_data(worm.value, 'drug_efficacy')

    logger.info("Start Monte Carlo simulation")
    df = pd.read_csv(path)
    mc_simulation = MCSimulation(df, worm)

    df_result = mc_simulation.total_simulation()
    df_result.to_csv("result.csv")

    print(df_result)


if __name__ == '__main__':
    main()
