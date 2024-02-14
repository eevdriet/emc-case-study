import math

import pandas as pd
from attrs import define

from emc.model.costs import Costs
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
    def total_cost(self):
        """
        Derive the total cost of following the given policy
        :return: Cost of the policy
        """
        survey_cost = self.__consumables_cost() + self.__personnel_cost() + self.__transportation_cost()

        cost = 0
        cost += survey_cost * sum(self.epi_surveys)
        cost += 2 * survey_cost * sum(self.drug_surveys)

        return cost

    def __consumables_cost(self):
        """
        Derive the policy costs associated with consumables
        :return: Cost of consumables
        """
        N = 430  # TODO: Get the number of total hosts (all age categories together
        samples = 1
        aliquots = 1
        aliquotCost = 1.37  # TODO: Assume single KK for now, duplicate KK = 1.51
        return 2 * N * samples * (Costs.EQUIPMENT + aliquots * aliquotCost)

    def __personnel_cost(self):
        """
        Derive the policy costs associated with personnel
        :return: Cost of personnel
        """
        return self.__days() * 4 * 22.50

    def __transportation_cost(self):
        """
        Derive the policy costs associated with transportation
        :return: Cost of transportation
        """
        return self.__days() * 90

    def __days(self):
        """
        Find the number of days required to conduct a survey
        :return: Days required to conduct a survey
        """
        N = 430  # TODO: Get the number of total hosts (all age categories together
        workers = 4  # Under assumption of single mobile field team: 1 nurse, three technicians
        timeAvailable = workers * 4 * 60 * 60  # In seconds
        c = 100  # TODO: Use average egg observations per time stamp
        timeProcessing = N * (15 + 67 + 9) + pow(10, 2.3896 + 0.0661 * math.log10(pow(c + 1, 2)))
        return 2 * math.ceil(timeProcessing / timeAvailable)

    def find_cost(self, scenario: Scenario) -> None:
        for simulation in scenario:
            for survey_index in self.surveys_i:
                if (True):  # TODO: get whether or not we have a signal of less than expected
                    # TODO: what if we do have DE survey
                    print('yes')
                else:
                    # TODO: what if we do not have DE survey
                    print('no')
