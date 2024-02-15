import math

import pandas as pd
from attrs import define

from emc.model.costs import Costs


@define
class Policy:
    # At which moments in time to conduct a survey or not
    deSurvey: list[bool] = [False] * 21
    epSurvey: list[bool] = [True] * 21

    @property
    def total_cost(self, de_survey: pd.DataFrame):
        survey_cost = self.__consumable(de_survey) + self.__personnel(de_survey) + self.__transportation(de_survey)

        cost = 0
        cost += 1 / 2 * survey_cost * sum(self.epSurvey)
        cost += survey_cost * sum(self.deSurvey)

        return cost

    def __consumable(self, de_survey: pd.DataFrame):
        N_baseline = (sum(de_survey['total_useful_tests']) + sum(de_survey['skipped_NaN_tests'])) / len(
            de_survey['total_useful_tests'])  # TODO: Is length of skipped = length useful??
        N_follow_up = sum(de_survey['total_useful_tests']) / len(de_survey['total_useful_tests'])
        baseline_costs = N_baseline * (Costs.EQUIPMENT + Costs.FIXED_COST + Costs.KATO_KATZ.get('single_sample'))
        follow_up_costs = N_follow_up * (
                    Costs.EQUIPMENT + Costs.FIXED_COST + 2 * Costs.KATO_KATZ.get('duplicate_sample'))
        return baseline_costs + follow_up_costs

    def __personnel(self, de_survey: pd.DataFrame):
        return self.__days(de_survey) * 4 * 22.50

    def __transportation(self, de_survey: pd.DataFrame):
        return self.__days(de_survey) * 90

    def __days(self, de_survey: pd.DataFrame):
        N = 430  # TODO: Get the number of total hosts (all age categories together
        workers = 4  # Under assumption of single mobile field team: 1 nurse, three technicians
        timeAvailable = workers * 4 * 60 * 60  # In seconds
        c = 100  # TODO: Use average egg observations per time stamp AND include duplicate KK
        countTime = 0
        for i in range(10):
            countTime += pow(10, 2.3896 + 0.0661 * math.log10(pow(c + 1, 2)))
            countTime += 2 * pow(10, 2.3896 + 0.0661 * math.log10(pow(c + 1, 2)))

        timeProcessing = 2 * N * (15 + 67 + 9) + countTime
        return math.ceil(timeProcessing / timeAvailable)
