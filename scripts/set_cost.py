import pandas as pd
import numpy as np
from math import isnan
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
            skipped_NaN_tests = len(pre) - len(post)
            true_a_pre = pre.mean(skipna=True)
            true_a_post = post.mean(skipna=True)

            costs = 0
            costs += self.__consumable(total_useful_tests, skipped_NaN_tests)

            days = self.__days(total_useful_tests, skipped_NaN_tests, true_a_pre, true_a_post)
            costs += self.__personnel(days)
            costs += self.__transportation(days)

            logger.info(f"Year {time}: {costs}")

    def __consumable(self, total_useful_tests: int, skipped_NaN_tests: int) -> float:
        """
        Calculate the consumable costs
        :param de_survey: Survey data to base costs on
        :param year: Year to schedule if any, otherwise take an average over all years
        :return: Consumable costs
        """
        # TODO: handle missing observations before calculating costs to avoid this error prevention below

        # Calculate costs
        N_baseline = total_useful_tests + skipped_NaN_tests
        N_follow_up = total_useful_tests
        baseline_costs = N_baseline * (Costs.EQUIPMENT + Costs.FIXED_COST + Costs.KATO_KATZ.get('single_sample'))
        follow_up_costs = N_follow_up * (
                Costs.EQUIPMENT + Costs.FIXED_COST + 2 * Costs.KATO_KATZ.get('duplicate_sample'))
        return baseline_costs + follow_up_costs

    def __personnel(self, days: int) -> float:
        """
        Calculate the personnel costs of a drug efficacy survey
        :param de_survey: Survey data to base costs on
        :param year: Year to schedule if any, otherwise take an average over all years
        :return: Personnel costs
        """
        return 4 * 22.5 * days

    def __transportation(self, days: int) -> int:
        """
        Calculate the transportation costs of a drug efficacy survey
        :param de_survey: Survey data to base costs on
        :param year: Year to schedule if any, otherwise take an average over all years
        :return: Transportation costs
        """
        return 90 * days

    def __days(self, total_useful_tests: int, skipped_NaN_tests: int, true_a_pre: float, true_a_post: float) -> int:
        """
        Calculate the number of days required to take a drug efficacy survey
        :param de_survey: Survey data to base the calculation on
        :param year: Year to schedule if any, otherwise take an average over all years
        :return: Survey days
        """
        # Set parameters
        workers = 4  # Under assumption of single mobile field team: 1 nurse, three technicians
        timeAvailable = workers * 4 * 60 * 60  # In seconds

        # Calculate costs
        N_baseline = total_useful_tests + skipped_NaN_tests
        N_follow_up = total_useful_tests
        counting_time = 0

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

    for worm in Worm:
        worm = worm.value

        # Load monitor age data
        path = Paths.data('csv') / f"{worm}_drug_efficacySC01SIM0001.csv"
        df = pd.read_csv(path)
        calculator.calculate_drug_cost(df)
