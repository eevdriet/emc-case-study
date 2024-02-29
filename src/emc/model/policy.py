import math
from math import isnan

import pandas as pd
import numpy as np
from typing import Generator, Iterable, Optional

from emc.model.costs import Costs
from emc.model.time_costs import Time_Costs
from emc.data.constants import *
from emc.util import first_or_mean


class Policy:
    """
    States at which point times a survey is conducted, which can be
    - an epidemiological survey
    - a drug efficacy survey (when a signal of drug resistance is suspected)
    """

    def __init__(self, epi_surveys: Iterable[bool]):
        # Store the surveys as tuples for hashing purposes
        self.epi_surveys = tuple(epi_surveys)
        assert len(self.epi_surveys) == N_YEARS

        # Initially schedule no drug efficacy surveys
        self.drug_surveys = (False,) * N_YEARS

    def calculate_cost(self, de_survey: pd.DataFrame) -> float:
        total_cost = 0

        # Calculate the cost of the drug efficacy surveys and add them to the total costs if relevant
        drug_surveys_costs = [self.__calculate_drug_cost(de_survey, year) for year in self.drug_time_points]

        drug_surveys = len(self.drug_time_points) != 0
        drug_data_complete = all(not isnan(cost) for cost in drug_surveys_costs)

        if not drug_surveys or not drug_data_complete:
            drug_surveys_costs = [self.__calculate_drug_cost(de_survey)]

        if drug_surveys and drug_data_complete:
            total_cost += sum(drug_surveys_costs)

        # Use half of the average drug efficacy survey cost for the epidemiological survey cost
        epi_surveys_costs = (1 / 2) * sum(drug_surveys_costs) / len(drug_surveys_costs) * sum(self.epi_surveys)
        total_cost += epi_surveys_costs

        return total_cost

    def __calculate_drug_cost(self, de_survey: pd.DataFrame, year: Optional[int] = None):
        """
        Calculate the cost of scheduling a drug efficacy survey in the given year
        :param de_survey: Data to base costs on
        :param year: Year to schedule if any, otherwise take an average over all years
        :return: Cost of scheduling the survey
        """
        costs = self.__consumable(de_survey, year) + self.__personnel(de_survey, year) + self.__transportation(
            de_survey, year)

        return costs

    def __hash__(self):
        return hash(self.epi_surveys)

    def __eq__(self, other):
        if not isinstance(other, Policy):
            return False

        return self.epi_surveys == other.epi_surveys

    def __len__(self):
        return sum(self.epi_surveys)

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}({self.epi_time_points})"

    def __getitem__(self, year: int) -> bool:
        assert 0 <= year < N_YEARS, "Year to schedule epidemiological survey needs to be valid"
        return self.epi_surveys[year]

    def __setitem__(self, year: int, do_survey: bool) -> "Policy":
        assert 0 <= year < N_YEARS, "Year to schedule epidemiological survey needs to be valid"

        # Create a new policy by setting the survey
        self.epi_surveys = self.epi_surveys[:year] + (do_survey,) + self.epi_surveys[year + 1:]

    @property
    def last_year(self):
        """
        Last year an epidemiological survey is taken
        :return: Year of the last epidemiological survey
        """
        # Default value in case no epidemiological survey is scheduled
        # This allows for scheduling a drug efficacy survey "one year after", i.e. at time 0
        if len(self.epi_time_points) < 1:
            return -1

        return self.epi_time_points[-1]

    @property
    def epi_time_points(self):
        return [time for time, do_survey in enumerate(self.epi_surveys) if do_survey]

    @property
    def drug_time_points(self):
        return [time for time, do_survey in enumerate(self.drug_surveys) if do_survey]

    def with_drug_survey(self) -> "Policy":
        year = self.last_year + 1
        assert 0 <= year < N_YEARS, "Year to schedule drug survey needs to be valid"

        # Create a new policy with an additional drug efficacy survey
        policy = Policy(self.epi_surveys)
        policy.drug_surveys = policy.drug_surveys[:year] + (True,) + policy.drug_surveys[year + 1:]
        return policy

    @property
    def sub_policies(self) -> Generator["Policy", None, None]:
        """
        Generate all sub-policies of the given policy
        :return:
        """
        for time in range(N_YEARS):
            if self.epi_surveys[time]:
                # Conduct all surveys up to and including the current year
                curr_years = self.epi_surveys[:time + 1]

                # Ignore any further years
                next_years = (False,) * (N_YEARS - time - 1)

                yield Policy(curr_years + next_years)

    def copy(self):
        return Policy(self.epi_surveys)

    @classmethod
    def __consumable(cls, de_survey: pd.DataFrame, year: Optional[int] = None):
        """
        Calculate the consumable costs
        :param de_survey: Survey data to base costs on
        :param year: Year to schedule if any, otherwise take an average over all years
        :return: Consumable costs
        """
        # Get data
        total_useful_tests = first_or_mean(de_survey, 'total_useful_tests', year)
        skipped_NaN_tests = first_or_mean(de_survey, 'skipped_NaN_tests', year)

        # TODO: handle missing observations before calculating costs to avoid this error prevention below
        if isnan(total_useful_tests) or isnan(skipped_NaN_tests):
            return np.nan

        # Calculate costs
        N_baseline = total_useful_tests + skipped_NaN_tests
        N_follow_up = total_useful_tests
        baseline_costs = N_baseline * (Costs.EQUIPMENT + Costs.FIXED_COST + Costs.KATO_KATZ.get('single_sample'))
        follow_up_costs = N_follow_up * (
                Costs.EQUIPMENT + Costs.FIXED_COST + 2 * Costs.KATO_KATZ.get('duplicate_sample'))
        return baseline_costs + follow_up_costs

    def __personnel(self, de_survey: pd.DataFrame, year: Optional[int] = None):
        """
        Calculate the personnel costs of a drug efficacy survey
        :param de_survey: Survey data to base costs on
        :param year: Year to schedule if any, otherwise take an average over all years
        :return: Personnel costs
        """
        return self.__days(de_survey, year) * 4 * 22.50

    def __transportation(self, de_survey: pd.DataFrame, year: Optional[int] = None) -> int:
        """
        Calculate the transportation costs of a drug efficacy survey
        :param de_survey: Survey data to base costs on
        :param year: Year to schedule if any, otherwise take an average over all years
        :return: Transportation costs
        """
        return self.__days(de_survey, year) * 90

    @classmethod
    def __days(cls, de_survey: pd.DataFrame, year: Optional[int] = None) -> int:
        """
        Calculate the number of days required to take a drug efficacy survey
        :param de_survey: Survey data to base the calculation on
        :param year: Year to schedule if any, otherwise take an average over all years
        :return: Survey days
        """
        # Get data
        total_useful_tests = first_or_mean(de_survey, 'total_useful_tests', year)
        skipped_NaN_tests = first_or_mean(de_survey, 'skipped_NaN_tests', year)
        true_a_pre = first_or_mean(de_survey, 'true_a_pre', year)
        true_a_post = first_or_mean(de_survey, 'true_a_post', year)

        # TODO: handle missing observations before calculating costs to avoid this error prevention below
        if any(isnan(var) for var in [total_useful_tests, skipped_NaN_tests, true_a_post, true_a_pre]):
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

def create_every_n_years_policy(every_n_years: int) -> Policy:
    """
    Create an initial policy to start the policy improvement from
    :param every_n_years: How often to schedule an epidemiological survey for the initial policy
    :return: Policy with an epidemiological survey every n years
    """
    # Perform the epi survey every n years
    tests = (True,) + (False,) * (every_n_years - 1)
    epi_surveys = tests * (N_YEARS // every_n_years) + tests[:N_YEARS % every_n_years]

    # Always (never) do a survey in the first (last) year
    epi_surveys = epi_surveys[:-1] + (False,)

    return Policy(epi_surveys)

def create_init_policy(policy_dict: dict) -> Policy:
    """
    Create an initial policy to start the policy improvement from
    :param every_n_years: How often to schedule an epidemiological survey for the initial policy
    :return: Policy with an epidemiological survey every n years
    """
    # Input + Always (never) do a survey in the first (last) year
    epi_surveys = (True,) + tuple(policy_dict.values()) + (False,)

    return Policy(epi_surveys)


if __name__ == '__main__':
    policy = create_init_policy(5)
    print(policy)
