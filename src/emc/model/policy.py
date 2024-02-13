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
    def total_cost(self):
        survey_cost = self.__consumable() + self.__personnel() + self.__transportation()

        cost = 0
        cost += survey_cost * sum(self.epSurvey)
        cost += 2 * survey_cost * sum(self.deSurvey)

        return cost

    def __consumable(self):
        N = 430  # TODO: Get the number of total hosts (all age categories together
        samples = 1
        aliquots = 1
        aliquotCost = 1.37  # TODO: Assume single KK for now, duplicate KK = 1.51
        return 2 * N * samples * (Costs.EQUIPMENT + aliquots * aliquotCost)

    def __personnel(self):
        return self.__days() * 4 * 22.50

    def __transportation(self):
        return self.__days() * 90

    def __days(self):
        N = 430
        workers = 4  # Under assumption of single mobile field team: 1 nurse, three technicians
        timeAvailable = workers * 4 * 60 * 60  # In seconds
        c = 100  # TODO: Find average for egg count or adjust to simulation
        timeProcessing = N * (15 + 67 + 9) + pow(10, 2.3896 + 0.0661 * math.log10(pow(c + 1, 2)))
        return 2 * math.ceil(timeProcessing / timeAvailable)
