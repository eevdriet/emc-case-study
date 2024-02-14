from attrs import define

from emc.model.scenario import Scenario, Simulation

@define
class Policy:
    # At which moments in time to conduct a survey or not
    surveys_b: list[bool]
    surveys_i: list[int]
    policy_cost: float

    def __init__(self, policy: list[int] = []):
        self.surveys = [False] * 21
        self.survey_index = policy
        self.policy_cost = 0

        for item in policy:
            self.surveys[item] == True

    def find_cost(self, scenario: Scenario) -> None:
        for simulation in scenario:
            for survey_index in self.surveys_i:
                if (True): # TODO: get whether or not we have a signal of less than expected
                    # TODO: 
                    print('yes')
                else:
                    # TODO: 
                    print('no')