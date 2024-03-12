from emc.log import setup_logger
from emc.model.policy import Policy
from emc.data.constants import *
from math import isnan
import numpy as np

logger = setup_logger(__name__)


class Objective(Enum):
    TOTAL_COSTS = 'total_costs'
    FINANCIAL_COSTS = 'financial_costs'
    RESPONSIVENESS = 'responsiveness'


class Score:
    __OBJECTIVE: Objective = Objective.TOTAL_COSTS
    __MULTIPLIER: float = 1.5
    __N_MAX_SURVEYS: int = 5

    __OBJECTIVE_FUNCS = {
    }

    """
    Aggregates all objectives that can be used to score a policy on its quality
    """

    def __init__(self, policy: Policy, n_simulations: int, n_wrong_classifications: int, responses: list[float],
                 sub_policy_costs: dict[Policy, dict[tuple[int, int], float]]):
        # Raw data
        self.policy = policy
        self.n_simulations = n_simulations
        self.n_wrong_classifications = n_wrong_classifications

        self.responses = responses
        self.sub_policy_cost = sub_policy_costs

        # Statistics
        self.total_sub_policy_costs = [cost for sub_policy in self.policy.sub_policies for cost in
                                       self.sub_policy_cost[sub_policy].values() if not isnan(cost)]
        self.nan_simulations = self.n_simulations - len(self.total_sub_policy_costs)

        self.avg_response = sum(self.responses) / len(self.responses)

        self.accuracy = 1 - (self.n_wrong_classifications / self.n_simulations)

        # Costs
        self.financial_costs = self.calculate_financial_costs()
        self.responsiveness_costs = self.avg_response * RESISTANCE_NOT_FOUND_COSTS
        self.accuracy_costs = (self.accuracy < 1 - MAX_MISCLASSIFICATION_FRACTION) * ACCURACY_VIOLATED_COSTS
        self.penalty_costs = self.calculate_penalty_costs()

        # Objectives
        self.score_funcs = {
            Objective.TOTAL_COSTS: self.calculate_total_costs,
            Objective.FINANCIAL_COSTS: self.calculate_financial_costs,
            Objective.RESPONSIVENESS: self.calculate_responsiveness_score,
        }

    def as_dict(self):
        score_str = f"score_{self.__OBJECTIVE.value}"

        def inf_str(val: float) -> str | float:
            return 'inf' if val == float('inf') else val

        return {
            'n_simulations': self.n_simulations,
            'n_wrong_classifications': self.n_wrong_classifications,
            'accuracy': self.accuracy,
            'avg_response': self.avg_response,
            'financial_costs': inf_str(self.financial_costs),
            'responsiveness_costs': inf_str(self.responsiveness_costs),
            'accuracy_costs': inf_str(self.accuracy_costs),
            score_str: inf_str(float(self))
        }

    @classmethod
    def create_missing(cls) -> "Score":
        """
        Missing object with infinite score
        :return: Missing score
        """
        times = (True,) + (False,) * (N_YEARS - 1)
        policy = Policy(times)
        costs = {policy: {(0, 0): float('inf')}}
        responses = [float('inf')]

        score = cls(policy=policy, n_simulations=1, n_wrong_classifications=0, responses=responses,
                    sub_policy_costs=costs)

        return score

    @classmethod
    def set_objective(cls, objective: Objective, **kwargs):
        """
        Set the objective to score on based on various arguments
        :param objective: Objective to score on
        :param kwargs: Key word arguments to modify the given objective
        """
        cls.__OBJECTIVE = objective

        if objective == Objective.RESPONSIVENESS:
            cls.__N_MAX_SURVEYS = kwargs.get('n_max_surveys', cls.__N_MAX_SURVEYS)
            cls.__MULTIPLIER = kwargs.get('multiplier', cls.__MULTIPLIER)

    def calculate_financial_costs(self):
        """
        Financial costs of the policy based on its survey moments
        :return: Financial costs of the score
        """
        if len(self.total_sub_policy_costs) <= 0:
            return float('inf')

        return sum(self.total_sub_policy_costs) / len(self.total_sub_policy_costs)

    def calculate_penalty_costs(self):
        """
        Penalty costs of the policy based on its responsiveness and accuracy
        :return: Penalty costs of the score
        """
        return self.responsiveness_costs + self.accuracy_costs

    def calculate_total_costs(self):
        """
        Total costs based on both the financial and penalty costs
        :return: Total costs of the score
        """
        return self.calculate_financial_costs() + self.calculate_penalty_costs()

    def calculate_responsiveness_score(self):
        if len(self.policy.epi_time_points) > self.__N_MAX_SURVEYS:
            return float('inf')

        responses = [response ** self.__MULTIPLIER for response in self.responses]
        return sum(responses) / len(responses)

    def __float__(self):
        """
        Actual score (value) that should be used to rank policies
        :return: Score value of the policy
        """
        # Accuracy should be valid for a valid score
        if self.accuracy < 1 - MAX_MISCLASSIFICATION_FRACTION:
            return float('inf')

        # Determine score based on the objective
        score_func = self.score_funcs[self.__OBJECTIVE]
        score = score_func()

        # Use helper function to get builtin float type instead of np.float
        return np.float64(score).item()

    def __lt__(self, other: "Score"):
        return float(self) < float(other)

    def __str__(self):
        score_str = self.__OBJECTIVE.value

        return f"""{self.policy}
- Total simulations (nan)   : {self.n_simulations} ({self.nan_simulations})
- Total responses (average) : {sum(self.responses)} ({self.avg_response})
- Total wrong (accuracy)    : {self.n_wrong_classifications} ({self.accuracy})
- Avg. financial costs      : {self.financial_costs}
- Avg. penalty   costs      : {self.penalty_costs} (response {self.responsiveness_costs} + accuracy {self.accuracy_costs})
---------------------------------------------------------
Total score ({score_str})   : {float(self)}
"""


if __name__ == '__main__':
    a = Score.__float__
    print(type(a))
