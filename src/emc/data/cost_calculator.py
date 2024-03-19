import math
from math import isnan
from typing import Optional

import numpy as np
import pandas as pd

from emc.data.constants import *
from emc.log import setup_logger
from emc.model.costs import Costs
from emc.model.time_costs import Time_Costs
from emc.util import Paths

logger = setup_logger(__name__)


class CostTechnique(Enum):
    FROM_INDIVIDUAL_HOSTS = 'hosts'
    FROM_AVERAGES = 'avg'


class CostCalculator:
    def __init__(self, worm: str, technique: CostTechnique = CostTechnique.FROM_INDIVIDUAL_HOSTS):
        """
        Constructor
        :param technique: What technique to use to derive costs
        """
        self.worm = worm
        self.technique = technique

        # Cost statistics
        self.total_useful_tests: Optional[int] = None
        self.skipped_NaN_tests: Optional[int] = None
        self.true_a_pre: Optional[float] = None
        self.true_a_post: Optional[float] = None

    def calculate_costs(self, host_df: pd.DataFrame) -> float:
        """
        Set the cost of scheduling a drug efficacy over all given years
        :param host_df: Data to set costs for
        """
        # Calculate statistics
        pre = host_df['pre']
        post = host_df['post']
        self.__calculate_statistics(pre, post)
        days = self.__calculate_days(pre, post)

        # Fixed costs (based on egg counts)
        costs = 0
        costs += self.__consumable()

        # Variable costs (based on egg counts)
        costs += self.__personnel(days)
        costs += self.__transportation(days)

        self.__reset_statistics()
        return costs

    def calculate_from_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all drug survey costs from a given data frame
        :param df: Data frame to calculate costs for
        :return: New data frame with drug survey costs added
        """

        def calculate_cost_from_hosts(group):
            # Retrieve the correct host data frame
            scenario, simulation, time = group.name
            path = Paths.host_data(self.worm, scenario, simulation)

            host_df = pd.read_csv(path)
            host_df = host_df[host_df['time'] == time]

            # Calculate the costs from the hosts
            costs = self.calculate_costs(host_df)
            group[f'cost_{self.technique.value}'] = costs
            del host_df

            logger.info(f"Costs for {scenario:02} {simulation:04} at t={time:02}: {costs}")
            return group

        # Calculate the cost per time point per simulation
        groups = df.groupby(['scenario', 'simulation', 'time'])
        groups = groups.apply(calculate_cost_from_hosts)
        groups = groups.reset_index(drop=True)

        return groups

    def __calculate_statistics(self, pre: pd.Series, post: pd.Series):
        """
        Calculate statistics that are commonly used for cost calculations
        :param pre: Pre-PC treatment egg counts
        :param post: Post-PC treatment egg counts
        """
        self.total_useful_tests = len(post) - post.isna().sum()
        self.skipped_NaN_tests = post.isna().sum()
        self.true_a_pre = pre.mean(skipna=True)
        self.true_a_post = post.mean(skipna=True)

        self.N_baseline = self.total_useful_tests + self.skipped_NaN_tests
        self.N_follow_up = self.total_useful_tests

    def __reset_statistics(self):
        """
        Reset statistics for more safety within different calculation functions
        """
        self.total_useful_tests = None
        self.skipped_NaN_tests = None
        self.true_a_pre = None
        self.true_a_post = None

        self.N_baseline = None
        self.N_follow_up = None

    def __calculate_days(self, pre: pd.Series, post: pd.Series):
        if self.technique == CostTechnique.FROM_AVERAGES:
            return self.__days_average()

        return self.__days_per_host(pre, post)

    def __calculate_costs_from_average(self, row: pd.Series):
        ...

    def __consumable(self) -> float:
        """
        Calculate the consumable costs
        :return: Consumable costs
        """
        # Determine costs
        baseline_costs = self.N_baseline * (Costs.EQUIPMENT + Costs.FIXED_COST + Costs.KATO_KATZ.get('single_sample'))
        follow_up_costs = self.N_follow_up * (
                Costs.EQUIPMENT + Costs.FIXED_COST + 2 * Costs.KATO_KATZ.get('duplicate_sample'))

        return baseline_costs + follow_up_costs

    @classmethod
    def __personnel(cls, days: int) -> float:
        """
        Calculate the personnel costs of a drug efficacy survey
        :param days: Number of days to base costs on
        :return: Personnel costs
        """
        return 4 * 22.5 * days

    @classmethod
    def __transportation(cls, days: int) -> int:
        """
        Calculate the transportation costs of a drug efficacy survey
        :param days: Number of days to base costs on
        :return: Transportation costs
        """
        return 90 * days

    def __days_per_host(self, pre: pd.Series, post: pd.Series) -> int:
        """
        Calculate the number of days required to take a drug efficacy survey
        :param pre: Pre-PC treatment egg counts
        :param post: Post-PC treatment egg counts
        :return: Survey days
        """
        # Set parameters
        workers = 4  # Under assumption of single mobile field team: 1 nurse, three technicians
        timeAvailable = workers * 4 * 60 * 60  # In seconds

        def countKK(series: pd.Series):
            """
            Calculate the Kato Katz
            :param series: Count data series to find the Kato Katz for
            :return: Kato Katz costs
            """
            return sum(Time_Costs.countKK(count) for count in series.dropna())

        count_pre = countKK(pre)
        count_post = 2 * countKK(post)

        time_pre = self.N_baseline * (Time_Costs.KATO_KATZ['demography'] + Time_Costs.KATO_KATZ.get('single_prep') +
                                      Time_Costs.KATO_KATZ.get('single_record')) + count_pre
        time_post = self.N_follow_up * (
                Time_Costs.KATO_KATZ.get('demography') + Time_Costs.KATO_KATZ.get('duplicate_prep') +
                Time_Costs.KATO_KATZ.get('duplicate_record')) + count_post

        return math.ceil((time_pre + time_post) / timeAvailable)

    def __days_average(self) -> float:
        """
        Calculate the number of days required to take a drug efficacy survey
        :return: Survey days
        """
        if isnan(self.true_a_pre) or isnan(self.true_a_post):
            return np.nan

        # Set parameters
        workers = 4  # Under assumption of single mobile field team: 1 nurse, three technicians
        timeAvailable = workers * 4 * 60 * 60  # In seconds

        # Calculate costs
        count_pre = Time_Costs.countKK(self.true_a_pre)
        count_post = 2 * Time_Costs.countKK(self.true_a_post)

        time_pre = self.N_baseline * (Time_Costs.KATO_KATZ['demography'] + Time_Costs.KATO_KATZ.get('single_prep') +
                                      Time_Costs.KATO_KATZ.get('single_record')) + count_pre
        time_post = self.N_follow_up * (
                Time_Costs.KATO_KATZ.get('demography') + Time_Costs.KATO_KATZ.get('duplicate_prep') +
                Time_Costs.KATO_KATZ.get('duplicate_record')) + count_post

        return math.ceil((time_pre + time_post) / timeAvailable)


if __name__ == '__main__':
    worm = Worm.ASCARIS.value
    calculator = CostCalculator(worm)

    # Calculate costs for a single host data frame
    scenario = 1
    simulation = 1
    time = 4

    path = Paths.host_data(worm, scenario, simulation)
    host_df = pd.read_csv(path)
    host_df = host_df[host_df['time'] == time]
    costs = calculator.calculate_costs(host_df)

    # Calculate costs for all scenarios
    path = Paths.worm_data(worm, 'drug_efficacy')
    df = pd.read_csv(path)
    new_df = calculator.calculate_from_df(df)
