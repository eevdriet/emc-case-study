from emc.log import setup_logger
from emc.model.policy import Policy
from emc.data.constants import *
from math import isnan
import numpy as np

logger = setup_logger(__name__)


class Score:
    """
    Aggregates all objectives that can be used to score a policy on its quality
    """

    def __init__(self, policy: Policy, n_simulations: int, n_wrong_classifications: int, latenesses: list[int],
                 sub_policy_costs: dict[Policy, dict[tuple[int, int], float]]):
        # Raw data
        self.policy = policy
        self.n_simulations = n_simulations
        self.n_wrong_classifications = n_wrong_classifications

        self.latenesses = latenesses
        self.sub_policy_cost = sub_policy_costs

        # Statistics
        self.total_sub_policy_costs = [cost for sub_policy in self.policy.sub_policies for cost in
                                       self.sub_policy_cost[sub_policy].values() if not isnan(cost)]
        self.nan_simulations = self.n_simulations - len(self.total_sub_policy_costs)

        self.avg_lateness = sum(self.latenesses) / len(self.latenesses)

        self.accuracy = 1 - (self.n_wrong_classifications / self.n_simulations)

        # Costs
        self.lateness_costs = self.avg_lateness * RESISTANCE_NOT_FOUND_COSTS
        self.accuracy_costs = (self.accuracy < 1 - MAX_MISCLASSIFICATION_FRACTION) * ACCURACY_VIOLATED_COSTS

    def to_json(self):
        return {
            'n_simulations': self.n_simulations,
            'n_wrong_classifications': self.n_wrong_classifications,
            'accuracy': self.accuracy,
            'avg_lateness': self.avg_lateness,
            'financial_costs': self.financial_costs,
            'lateness_costs': self.lateness_costs,
            'accuracy_costs': self.accuracy_costs,
            'total_costs': float(self)
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
        latenesses = [0]

        score = cls(policy=policy, n_simulations=1, n_wrong_classifications=0, latenesses=latenesses,
                    sub_policy_costs=costs)

        return score

    @property
    def financial_costs(self):
        if len(self.total_sub_policy_costs):
            return sum(self.total_sub_policy_costs) / len(self.total_sub_policy_costs)

        return float('inf')

    @property
    def penalty_costs(self):
        return self.lateness_costs + self.accuracy_costs

    @property
    def total_costs(self):
        return self.financial_costs + self.penalty_costs

    def __float__(self):
        return np.float64(self.total_costs).item()

    def __str__(self):
        return f"""{self.policy}
- Total simulations (nan)  : {self.n_simulations} ({self.nan_simulations})
- Total lateness (average) : {sum(self.latenesses)} ({self.avg_lateness})
- Total wrong (accuracy)   : {self.n_wrong_classifications} ({self.accuracy})
- Avg. financial costs     : {self.financial_costs}
- Avg. penalty   costs     : {self.penalty_costs} (lateness {self.lateness_costs} + accuracy {self.accuracy_costs})
---------------------------------------------------------
Total score                : {float(self)}
"""
