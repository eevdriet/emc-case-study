import math

import pandas as pd
from attrs import define

from emc.model.costs import Costs
from emc.model.time import Time
from emc.model.scenario import Scenario

@define
class Policy:
    """
    States at which point times a survey is conducted, which can be
    - an epidemiological survey
    - a drug efficacy survey (when a signal of drug resistance is suspected)
    """

    # At which moments in time to conduct a drug efficacy survey or not
    drug_surveys: list[bool] = [False] * 21

    # At which moments in time to conduct an epidemiological survey or not
    epi_surveys: list[bool] = [True] * 21

    surveys_b: list[bool] = [True] * 21
    surveys_i: list[int] = [1] * 21

    @property
    def total_cost(self, de_survey: pd.DataFrame):
        survey_cost = self.__consumable(de_survey) + self.__personnel(de_survey) + self.__transportation(de_survey)

        cost = 1 / 2 * survey_cost * sum(self.epSurvey)
        cost += survey_cost * sum(self.deSurvey)

        return cost

    def __consumable(self, de_survey: pd.DataFrame):
        N_baseline = de_survey['total_useful_tests'] + de_survey['skipped_NaN_tests']
        N_follow_up = de_survey['total_useful_tests']
        baseline_costs = N_baseline * (Costs.EQUIPMENT + Costs.FIXED_COST + Costs.KATO_KATZ.get('single_sample'))
        follow_up_costs = N_follow_up * (
                Costs.EQUIPMENT + Costs.FIXED_COST + 2 * Costs.KATO_KATZ.get('duplicate_sample'))
        return baseline_costs + follow_up_costs

    def __personnel(self, de_survey: pd.DataFrame):
        return self.__days(de_survey) * 4 * 22.50

    def __transportation(self, de_survey: pd.DataFrame):
        return self.__days(de_survey) * 90

    def __days(self, de_survey: pd.DataFrame):
        workers = 4  # Under assumption of single mobile field team: 1 nurse, three technicians
        timeAvailable = workers * 4 * 60 * 60  # In seconds
        N_baseline = de_survey['total_useful_tests'] + de_survey['skipped_NaN_tests']
        N_follow_up = de_survey['total_useful_tests']
        c_pre = de_survey['true_a_pre']  # TODO: Use average egg observations per time stamp AND include duplicate KK
        c_post = de_survey['true_a_post']  # TODO: This is true number of eggs in individual, aliquots is on observed
        count_pre = self.__countKK(self, c_pre)
        count_post = self.__countKK(self, 2 * c_post)
        time_pre = N_baseline * (Time.KATO_KATZ.get('demography') + Time.KATO_KATZ.get('single_prep') +
                                 Time.KATO_KATZ.get('single_record')) + count_pre
        time_post = N_follow_up * (Time.KATO_KATZ.get('demography') + Time.KATO_KATZ.get('duplicate_prep') +
                                  Time.KATO_KATZ.get('duplicate_record')) + count_post
        return math.ceil((time_pre + time_post) / timeAvailable)

    # TODO: Add to time class if possible
    def __countKK(self, count: int):
        return pow(10, 2.3896 + 0.0661 * pow(math.log10(count + 1), 2))

    def __countMF(self, count: int):
        return pow(10, 2.5154 + 0.0661 * pow(math.log10(count + 1), 2))

    def __countFEC(self, count: int):
        return pow(10, 1.8349 + 0.1731 * pow(math.log10(count + 1), 2))