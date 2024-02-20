import math

import pandas as pd
from typing import Generator, Iterable

from emc.model.costs import Costs
from emc.model.time_costs import Time_Costs
from emc.data.constants import *


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

        # Find all time points for which the epidemiological surveys are scheduled
        self.time_points = [time for time, do_survey in enumerate(self.epi_surveys) if do_survey]

        # Initially schedule no drug efficacy surveys
        self.drug_surveys = (False,) * N_YEARS

    def calculate_cost(self, de_survey: pd.DataFrame):
        survey_cost = self.__consumable(de_survey) + self.__personnel(de_survey) + self.__transportation(de_survey)

        cost = 0
        cost += (1 / 2) * survey_cost * sum(self.epi_surveys)
        cost += survey_cost * sum(self.drug_surveys)

        return cost

    def __hash__(self):
        return hash(self.epi_surveys)

    def __len__(self):
        return sum(self.epi_surveys)

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}({self.time_points})"

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

    @classmethod
    def __consumable(cls, de_survey: pd.DataFrame):
        N_baseline = de_survey['total_useful_tests'] + de_survey['skipped_NaN_tests']
        N_follow_up = de_survey['total_useful_tests']
        baseline_costs = N_baseline * (Costs.EQUIPMENT + Costs.FIXED_COST + Costs.KATO_KATZ.get('single_sample'))
        follow_up_costs = N_follow_up * (
                Costs.EQUIPMENT + Costs.FIXED_COST + 2 * Costs.KATO_KATZ.get('duplicate_sample'))
        return baseline_costs + follow_up_costs

    def __personnel(self, de_survey: pd.DataFrame):
        return self.__days(de_survey) * 4 * 22.50

    def __transportation(self, de_survey: pd.DataFrame) -> int:
        return self.__days(de_survey) * 90

    @classmethod
    def __days(cls, de_survey: pd.DataFrame) -> int:
        workers = 4  # Under assumption of single mobile field team: 1 nurse, three technicians
        timeAvailable = workers * 4 * 60 * 60  # In seconds
        N_baseline = de_survey['total_useful_tests'] + de_survey['skipped_NaN_tests']
        N_follow_up = de_survey['total_useful_tests']
        c_pre = de_survey['true_a_pre']  # TODO: Use average egg observations per time stamp AND include duplicate KK
        c_post = de_survey['true_a_post']  # TODO: This is true number of eggs in individual, aliquots is on observed
        count_pre = Time_Costs.countKK(c_pre)
        count_post = Time_Costs.countKK(2 * c_post)
        time_pre = N_baseline * (Time_Costs.KATO_KATZ['demography'] + Time_Costs.KATO_KATZ.get('single_prep') +
                                 Time_Costs.KATO_KATZ.get('single_record')) + count_pre
        time_post = N_follow_up * (Time_Costs.KATO_KATZ.get('demography') + Time_Costs.KATO_KATZ.get('duplicate_prep') +
                                   Time_Costs.KATO_KATZ.get('duplicate_record')) + count_post
        return math.ceil((time_pre + time_post) / timeAvailable)


if __name__ == '__main__':
    import inspect

    print(inspect.getsource(Policy.__hash__))
