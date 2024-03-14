import math
from math import isnan
import random

import pandas as pd
import numpy as np
from typing import Generator, Iterable, Optional

from emc.model.costs import Costs
from emc.model.time_costs import Time_Costs
from emc.data.constants import *
from emc.util import first_or_mean

random.seed(SEED)

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
        assert self.epi_surveys[0], "Should conduct epidemiological survey in the first year"
        assert not self.epi_surveys[N_YEARS - 1], "Should not conduct epidemiological survey in the final year"

        # Initially schedule no drug efficacy surveys
        self.drug_surveys = (False,) * N_YEARS

    def __eq__(self, other: "Policy") -> bool:
        """
        Check if two policies are equal based on their epi_surveys attribute only.
        """
        if not isinstance(other, Policy):
            return False

        return self.epi_surveys == other.epi_surveys
    
    @classmethod
    def from_timepoints(cls, time_points: list[int]) -> "Policy":
        assert min(time_points) >= 0, "Minimum time should be 0"
        assert max(time_points) < N_YEARS, f"Maximum time should be {N_YEARS - 1}"

        epi_surveys = tuple(time in time_points for time in range(N_YEARS))
        return Policy(epi_surveys)

    @classmethod
    def from_every_n_years(cls, n: int) -> "Policy":
        """
        Create an initial policy to start the policy improvement from
        :param n: How often to schedule an epidemiological survey for the initial policy
        :return: Policy with an epidemiological survey every n years
        """
        # Perform the epi survey every n years
        tests = (True,) + (False,) * (n - 1)
        epi_surveys = tests * (N_YEARS // n) + tests[:N_YEARS % n]

        # Always (never) do a survey in the first (last) year
        epi_surveys = epi_surveys[:-1] + (False,)

        return Policy(epi_surveys)
    
    def perturbe(self) -> "Policy":
        input_list = list(self.epi_surveys)
        true_count = input_list.count(True)

        # Increase the number of True values by one
        new_true_count = true_count + 1
        if new_true_count >= len(input_list):
            return Policy(self.epi_surveys)

        def is_true_within_one(index, list_to_check):
            if index > 0 and list_to_check[index - 1]:
                return True
            if index < len(list_to_check) - 1 and list_to_check[index + 1]:
                return True
            return False
        
        # Function to find all possible positions for a new True value
        def find_possible_positions(list_to_check):
            possible_positions = []
            for i in range(1, len(list_to_check) - 1):  # Exclude the first and last elements
                if not list_to_check[i] and is_true_within_one(i, list_to_check):
                    possible_positions.append(i)
            return possible_positions
        
        # Set all elements to False initially, except the first one
        new_epi_surveys = [False] * len(input_list)
        new_epi_surveys[0] = True
        new_epi_surveys[-1] = False  # Ensure the last element remains False

        # Place new True values in the result list
        for _ in range(new_true_count - 1):  # Subtract 1 as the first True is already placed
            possible_positions = find_possible_positions(input_list)
            if not possible_positions:
                return Policy(self.epi_surveys)
            index = random.choice(possible_positions)
            new_epi_surveys[index] = True
            input_list[index] = True  # Update input_list to reflect the new True value

        return Policy(tuple(new_epi_surveys))

    def calculate_cost(self, de_survey: pd.DataFrame, allow_average: bool = True) -> float:
        """
        Calculate the cost of the policy for the given simulation data
        :param de_survey: Data from a simulation
        :param allow_average: Whether cost is allowed to be calculated based on average over all policy years
        :return: Cost of the policy
        """

        total_cost = 0

        # Calculate the cost of the drug efficacy surveys and add them to the total costs if relevant
        drug_surveys_costs = [self.calculate_drug_cost(de_survey, year) for year in self.drug_time_points]

        drug_surveys = len(self.drug_time_points) != 0
        drug_data_complete = all(not isnan(cost) for cost in drug_surveys_costs)

        # Treat no drug surveys as missing costs
        if not drug_surveys:
            drug_surveys_costs = [np.nan]

        # Base the costs on the average over all policy years if allowed and current costs invalid
        if allow_average and (not drug_surveys or not drug_data_complete):
            drug_surveys_costs = [self.calculate_drug_cost(de_survey)]

        if drug_surveys and drug_data_complete:
            total_cost += sum(drug_surveys_costs)

        # Use half of the average drug efficacy survey cost for the epidemiological survey cost
        epi_surveys_costs = (1 / 2) * sum(drug_surveys_costs) / len(drug_surveys_costs) * sum(self.epi_surveys)
        total_cost += epi_surveys_costs

        return total_cost

    def calculate_drug_cost(self, de_survey: pd.DataFrame, year: Optional[int] = None) -> float:
        """
        Calculate the cost of scheduling a drug efficacy survey in the given year
        :param de_survey: Data to base costs on
        :param year: Year to schedule if any, otherwise take an average over all years
        :return: Cost of scheduling the survey
        """
        # Compute average if no specific year is given
        if year is None:
            return de_survey['cost'].mean(skipna=True)

        # Otherwise verify the year is valid and get its cost
        series = de_survey.loc[de_survey['time'] == year, 'cost']
        if series.empty:
            return np.nan

        return series.iloc[0]

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

    def __setitem__(self, year: int, do_survey: bool):
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


if __name__ == '__main__':
    policy = Policy.from_every_n_years(5)
    print(policy)
    policy.perturbe()
    print(policy)
