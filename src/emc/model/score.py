from emc.log import setup_logger
from emc.model.policy import Policy
from emc.data.constants import *
from math import isnan
import numpy as np

logger = setup_logger(__name__)

class ScoreType(Enum):
    TOTAL_COSTS = 'total_costs'
    FINANCIAL_COSTS = 'financial_costs'
    RESPONSIVENESS = 'responsiveness'

class Score:
    """
    Aggregates all objectives that can be used to score a policy on its quality
    """

    def __init__(self, policy: Policy, n_simulations: int, n_false_positives: int, n_false_negatives: int, responses: list[int],
                 sub_policy_costs: dict[Policy, dict[tuple[int, int], float]], score_type=ScoreType.TOTAL_COSTS):
        # Raw data
        self.policy = policy
        self.n_simulations = n_simulations
        self.n_false_positives = n_false_positives
        self.n_false_negatives = n_false_negatives

        self.responses = responses
        self.sub_policy_cost = sub_policy_costs

        # Statistics
        self.total_sub_policy_costs = [cost for sub_policy in self.policy.sub_policies for cost in
                                       self.sub_policy_cost[sub_policy].values() if not isnan(cost)]
        self.nan_simulations = self.n_simulations - len(self.total_sub_policy_costs)

        self.avg_response = sum(self.responses) / len(self.responses)

        self.accuracy = 1 - ((self.n_false_positives + self.n_false_negatives) / self.n_simulations)

        # Costs
        self.responsiveness_costs = self.avg_response * RESISTANCE_NOT_FOUND_COSTS
        self.accuracy_costs = (self.accuracy < 1 - MAX_MISCLASSIFICATION_FRACTION) * ACCURACY_VIOLATED_COSTS

        self.score_type = score_type

    def as_dict(self):
        return {
            'n_simulations': self.n_simulations,
            'n_false_positives': self.n_false_positives,
            'n_false_negatives': self.n_false_negatives,
            'accuracy': self.accuracy,
            'avg_lateness': self.avg_response,
            'financial_costs': self.financial_costs,
            'lateness_costs': self.responsiveness_costs,
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
        latenesses = [float('inf')]

        score = cls(policy=policy, n_simulations=1, n_false_positives=0, n_false_negatives=0, responses=latenesses,
                    sub_policy_costs=costs)

        return score

    @property
    def financial_costs(self):
        """
        Financial costs of the policy based on its survey moments
        :return: Financial costs of the score
        """
        if len(self.total_sub_policy_costs):
            return sum(self.total_sub_policy_costs) / len(self.total_sub_policy_costs)

        return float('inf')

    @property
    def penalty_costs(self):
        """
        Penalty costs of the policy based on its responsiveness and accuracy
        :return: Penalty costs of the score
        """
        return self.responsiveness_costs + self.accuracy_costs

    @property
    def total_costs(self):
        """
        Total costs based on both the financial and penalty costs
        :return: Total costs of the score
        """
        return self.financial_costs + self.penalty_costs

    def __float__(self):
        if self.score_type == ScoreType.TOTAL_COSTS:
            return self._calculate_total_costs_score()
        elif self.score_type == ScoreType.FINANCIAL_COSTS:
            return self._calculate_financial_costs_score()
        elif self.score_type == ScoreType.RESPONSIVENESS:
            return self._calculate_responsiveness_score()
        else:
            raise ValueError("Invalid score calculation method")
        
    def _calculate_total_costs_score(self):
        if self.accuracy < 0.80:
            return float('inf')
    
        score = self.total_costs
        return np.float64(score).item()
    
    def _calculate_financial_costs_score(self):
        if self.accuracy < 0.80:
            return float('inf')
        
        score = self.financial_costs
        return np.float64(score).item()

    def _calculate_responsiveness_score(self):
        if self.accuracy < 0.80:
            return float('inf')
        if len(self.policy.epi_time_points) > 5:
            return float('inf')
    
        responses = [response ** 1.5 for response in self.responses]
        
        score = sum(responses) / len(responses)
        return np.float64(score).item()

    def __lt__(self, other: "Score"):
        return float(self) < float(other)

    def __str__(self):
        return f"""{self.policy}
- Total simulations (nan)        : {self.n_simulations} ({self.nan_simulations})
- Total responsiveness (average)      : {sum(self.responses)} ({self.avg_response})
- Total wrong (accuracy, FP, FN) : {self.n_false_negatives + self.n_false_positives} ({self.accuracy}, {self.n_false_positives}, {self.n_false_negatives})
- Avg. financial costs           : {self.financial_costs}
- Avg. penalty   costs           : {self.penalty_costs} (response {self.responsiveness_costs} + accuracy {self.accuracy_costs})
---------------------------------------------------------
Total score                      : {float(self)}
"""
