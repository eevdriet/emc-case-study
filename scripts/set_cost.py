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

    def calculate_drug_cost(self, df: pd.DataFrame):
        """
        Calculate the cost of scheduling a drug efficacy survey in the given year
        :param de_survey: Data to base costs on
        :param year: Year to schedule if any, otherwise take an average over all years
        :return: Cost of scheduling the survey
        """
        # Logging
        scenario = df['scenario']
        simulation = df['simulation']
        logger.info(f"{scenario.iloc[0]} {simulation.iloc[0]}")

        # Parameters
        name = 'drug_efficacy.time'
        for time in df[name].unique():
            pre = df.loc[df[name] == time, 'drug_efficacy.pre']
            post = df.loc[df[name] == time, 'drug_efficacy.post']

            total_useful_tests = len(pre)
            skipped_NaN_tests = post.isna().sum()
            true_a_pre = pre.mean(skipna=True)
            true_a_post = post.mean(skipna=True)

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

            # logger.info(f"Year {time}: {costs} ({days}) <-> {costs2} ({days2})")

    @classmethod
    def __consumable(cls, pre: pd.Series, post: pd.Series) -> float:
        """
        Calculate the consumable costs
        :param pre: Survey data to base costs on
        :param year: Year to schedule if any, otherwise take an average over all years
        :return: Consumable costs
        """
        # TODO: handle missing observations before calculating costs to avoid this error prevention below
        total_useful_tests = len(pre)
        skipped_NaN_tests = post.isna().sum()

        # Calculate costs
        N_baseline = total_useful_tests + skipped_NaN_tests
        N_follow_up = total_useful_tests
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
        total_useful_tests = len(pre)
        skipped_NaN_tests = post.isna().sum()
        true_a_pre = pre.mean(skipna=True)
        true_a_post = post.mean(skipna=True)

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
        total_useful_tests = len(pre)
        skipped_NaN_tests = len(pre) - len(post)
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
    calculator = CostCalculator(save=True)

    for worm in [Worm.ASCARIS]:
        worm = worm.value

        for scenario in range(1, N_SCENARIOS + 1):
            for simulation in range(1, N_SIMULATIONS + 1):
                # Load monitor age data
                path = Paths.data('csv') / f"{worm}_drug_efficacySC{scenario:02d}SIM{simulation:04d}.csv"
                df = pd.read_csv(path)
                calculator.calculate_drug_cost(df)

        print(calculator.days)
        print(calculator.days2)
