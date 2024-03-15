import pandas as pd
import numpy as np
from scipy.stats import gamma
from scipy.stats import poisson
import sys as sys

from emc.data.constants import *
# from emc.data.cost_calculator import CostCalculator
from emc.model.simulation import Simulation
from emc.model.policy import Policy
from emc.util import Paths
from emc.log import setup_logger

logger = setup_logger(__name__)


class MonteCarlo:
    """
    Performs a Monte Carlo simulation of the observed egg counts, both pre and post drug treatment
    """
    # Parameters
    __SHAPE_SLIDE = sys.maxsize
    __N_SIMULATIONS = 5000

    def __init__(self, worm: str):
        self.__SHAPE_DAY = 0.510 if worm == 'ascaris' else 1.000

    def run(self, simulation: Simulation, policy: Policy):
        """
        Run the simulation and update associated costs
        :param simulation: Simulation to run for
        """
        # Determine in which year to simulate
        year = policy.last_year + 1

        # Perform simulation of the pre/post egg counts in the unmerged data
        host_df = simulation.unmerged_drug_efficacy_s
        host_df = host_df.loc[host_df['time'] == year]
        self.simulate(host_df)

        # Update associated attributes
        df = simulation.drug_efficacy_s
        df = df.loc[df['time'] == year]

        # - ERR
        pre = host_df['pre'].mean(skipna=True)
        post = host_df['post'].mean(skipna=True)
        df['ERR'] = 1 - post / pre

        # - Costs
        df['cost'] = 4  # CostCalculator.update(simulation)

    def simulate(self, df: pd.DataFrame):
        """
        Perform the actual simulation based on the pre/post egg count data
        :param df: Data to modify by simulation
        :return:
        """
        df['pre'] = df['pre'].apply(self.__simulate_count_ids)
        df['post'] = df['post'].apply(self.__simulate_count_ids)

    def __simulate_count_ids(self, mu_i: float):
        """
        Simulate the egg counts based on an initial shape parameter
        :param mu_i: Initial shape parameter
        :return: Simulated egg counts
        """
        # Handle missing data
        if np.isnan(mu_i):
            return mu_i

        logger.debug(f"\t- mu_i: {mu_i}")

        mu_id = gamma.rvs(self.__SHAPE_DAY, scale=mu_i / self.__SHAPE_DAY)
        logger.debug(f"\t- mu_id: {mu_id}")

        mu_ids = gamma.rvs(self.__SHAPE_SLIDE, scale=mu_id / self.__SHAPE_SLIDE)
        logger.debug(f"\t- mu_ids: {mu_ids}")

        count = poisson.rvs(mu_ids)
        logger.debug(f"\t- count: {count}")

        return count


def main():
    # Setup scenario
    worm = Worm.ASCARIS.value
    scenario = 1
    simulation = 1

    # Load data
    path = Paths.data('csv') / f'{worm}_drug_efficacySC{scenario:02}SIM{simulation:04}.feather'
    df = pd.read_feather(path.with_suffix('.feather'))

    # Run simulation
    monte_carlo = MonteCarlo(worm)
    monte_carlo.simulate(df)


if __name__ == '__main__':
    main()
