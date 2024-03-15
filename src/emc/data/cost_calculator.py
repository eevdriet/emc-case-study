import pandas as pd
import numpy as np
from math import isnan
from collections import Counter
import math

from emc.util import Paths
from emc.data.constants import *
from emc.log import setup_logger
from emc.model.costs import Costs
from emc.model.time_costs import Time_Costs

logger = setup_logger(__name__)


class CostCalculator:
    ...

    def __init__(self, save: True):
        """
        Constructor
        :param save: Whether to save the costs in the modified data frame back to memory
        """
        self.save = save
        self.days = Counter()
        self.days2 = Counter()

    @classmethod
    def calculate_costs(cls, df: pd.DataFrame) -> float:
        """
        Set the cost of scheduling a drug efficacy over all given years
        :param df: Data to set costs for
        """
        pre = df['pre']
        post = df['pre']

        total_useful_tests = len(post) - post.isna().sum()
        skipped_NaN_tests = post.isna().sum()
        true_a_pre = pre.mean(skipna=True)
        true_a_post = post.mean(skipna=True)

        return float('inf')

    def calculate_drug_cost(self, df: pd.DataFrame, use_averages: bool = True):
        """
        Calculate the cost of scheduling a drug efficacy survey in the given year
        :param de_survey: Data to base costs on
        :param year: Year to schedule if any, otherwise take an average over all years
        :return: Cost of scheduling the survey
        """
        if use_averages:
            df['cost'] = df.apply(self.__calculate_drug_cost, axis=1)
        # Logging
        scenario = df['scenario'].iloc[0]
        simulation = df['simulation'].iloc[0]
        # logger.info(f"{scenario} {simulation}")

        for scenario in range(1, N_SCENARIOS + 1):
            logger.info(f"Scenario {scenario}")
            for simulation in range(1, N_SIMULATIONS + 1):
                # Load monitor age data
                path = Paths.data('csv') / f"{worm}_drug_efficacySC{scenario:02d}SIM{simulation:04d}.csv"
                df = pd.read_csv(path)

        # Parameters
        for time in df['time'].unique():
            pre = df.loc[df['time'] == time, 'pre']
            post = df.loc[df['time'] == time, 'post']

            # Calculate using averages
            costs = 0
            costs += self.__consumable(pre, post)

            days = self.__days_average(pre, post)
            self.days[days] += 1
            costs += self.__personnel(days)
            costs += self.__transportation(days)

            # Calculate using hosts
            costs2 = 0
            costs2 += self.__consumable(pre, post)

            days2 = self.__days_per_host(pre, post)
            self.days2[days2] += 1
            costs2 += self.__personnel(days2)
            costs2 += self.__transportation(days2)

            df3 = df2[(df2['scenario'] == scenario) & (df2['simulation'] == simulation) & (df2['time'] == time)]
            cost_old = df3['cost'].iloc[0]

            if isnan(costs) and isnan(cost_old):
                continue
            if abs(costs - cost_old) > 1e-2:
                logger.info(f"{scenario} {simulation} {time}: {costs} != {cost_old}")

    def __calculate_costs_from_average(self, row: pd.Series):


    @classmethod
    def calculate(self, pre: pd.DataFrame, post: pd.DataFrame):
        costs = 0
        costs += self.__consumable(pre, post)

        days = self.__days_average(pre, post)
        self.days[days] += 1
        costs += self.__personnel(days)
        costs += self.__transportation(days)

        # Calculate using hosts
        costs2 = 0
        costs2 += self.__consumable(pre, post)

        days2 = self.__days_per_host(pre, post)
        self.days2[days2] += 1
        costs2 += self.__personnel(days2)
        costs2 += self.__transportation(days2)

    @classmethod
    def __consumable(cls, total_useful_tests: int, skipped_NaN_tests: int) -> float:
        """
        Calculate the consumable costs
        :param pre: Survey data to base costs on
        :param year: Year to schedule if any, otherwise take an average over all years
        :return: Consumable costs
        """
        # Determine number of hosts
        N_baseline = total_useful_tests + skipped_NaN_tests
        N_follow_up = total_useful_tests

        # Determine costs
        baseline_costs = N_baseline * (Costs.EQUIPMENT + Costs.FIXED_COST + Costs.KATO_KATZ.get('single_sample'))
        follow_up_costs = N_follow_up * (
                Costs.EQUIPMENT + Costs.FIXED_COST + 2 * Costs.KATO_KATZ.get('duplicate_sample'))

        return baseline_costs + follow_up_costs

    @classmethod
    def __personnel(cls, days: int) -> float:
        """
        Calculate the personnel costs of a drug efficacy survey
        :param de_survey: Survey data to base costs on
        :param year: Year to schedule if any, otherwise take an average over all years
        :return: Personnel costs
        """
        return 4 * 22.5 * days

    @classmethod
    def __transportation(cls, days: int) -> int:
        """
        Calculate the transportation costs of a drug efficacy survey
        :param de_survey: Survey data to base costs on
        :param year: Year to schedule if any, otherwise take an average over all years
        :return: Transportation costs
        """
        return 90 * days

    @classmethod
    def __days_per_host(cls, pre: pd.Series, post: pd.Series) -> int:
        """
        Calculate the number of days required to take a drug efficacy survey
        :param de_survey: Survey data to base the calculation on
        :param year: Year to schedule if any, otherwise take an average over all years
        :return: Survey days
        """
        # Calculate statistics
        total_useful_tests = len(post) - post.isna().sum()
        skipped_NaN_tests = post.isna().sum()

        # Set parameters
        workers = 4  # Under assumption of single mobile field team: 1 nurse, three technicians
        timeAvailable = workers * 4 * 60 * 60  # In seconds

        # Calculate costs
        N_baseline = total_useful_tests + skipped_NaN_tests
        N_follow_up = total_useful_tests

        def countKK(series: pd.Series):
            """
            Calculate the Kato Katz
            :param series: Count data series to find the Kato Katz for
            :return: Kato Katz costs
            """
            return sum(Time_Costs.countKK(count) for count in series.dropna())

        count_pre = countKK(pre)
        count_post = 2 * countKK(post)

        time_pre = N_baseline * (Time_Costs.KATO_KATZ['demography'] + Time_Costs.KATO_KATZ.get('single_prep') +
                                 Time_Costs.KATO_KATZ.get('single_record')) + count_pre
        time_post = N_follow_up * (Time_Costs.KATO_KATZ.get('demography') + Time_Costs.KATO_KATZ.get('duplicate_prep') +
                                   Time_Costs.KATO_KATZ.get('duplicate_record')) + count_post
        return math.ceil((time_pre + time_post) / timeAvailable)

    @classmethod
    def __days_average(cls, pre: pd.Series, post: pd.Series) -> int:
        """
        Calculate the number of days required to take a drug efficacy survey
        :param de_survey: Survey data to base the calculation on
        :param year: Year to schedule if any, otherwise take an average over all years
        :return: Survey days
        """
        # Calculate statistics
        total_useful_tests = len(post) - post.isna().sum()
        skipped_NaN_tests = post.isna().sum()
        true_a_pre = pre.mean(skipna=True)
        true_a_post = post.mean(skipna=True)

        if isnan(true_a_pre) or isnan(true_a_post):
            return np.nan

        # Set parameters
        workers = 4  # Under assumption of single mobile field team: 1 nurse, three technicians
        timeAvailable = workers * 4 * 60 * 60  # In seconds

        # Calculate costs
        N_baseline = total_useful_tests + skipped_NaN_tests
        N_follow_up = total_useful_tests

        c_pre = true_a_pre  # TODO: Use average egg observations per time stamp AND include duplicate KK
        c_post = true_a_post  # TODO: This is true number of eggs in individual, aliquots is on observed

        count_pre = Time_Costs.countKK(c_pre)
        count_post = Time_Costs.countKK(2 * c_post)
        time_pre = N_baseline * (Time_Costs.KATO_KATZ['demography'] + Time_Costs.KATO_KATZ.get('single_prep') +
                                 Time_Costs.KATO_KATZ.get('single_record')) + count_pre
        time_post = N_follow_up * (Time_Costs.KATO_KATZ.get('demography') + Time_Costs.KATO_KATZ.get('duplicate_prep') +
                                   Time_Costs.KATO_KATZ.get('duplicate_record')) + count_post
        return math.ceil((time_pre + time_post) / timeAvailable)


if __name__ == '__main__':
    calculator = CostCalculator()

    worm = Worm.ASCARIS.value
    scenario = 1
    simulation = 4
    path = Paths.data('csv') / f'{worm}_drug_efficacySC{scenario:02}SIM{simulation:04}.feather'
    path = Paths.worm_data(worm, 'drug_efficacy')
    df_merged = pd.read_csv(path)

    calculator.calculate_drug_cost(df_merged, use_averages=True)

    print(calculator.days)
    print(calculator.days2)
