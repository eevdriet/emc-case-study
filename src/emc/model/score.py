from attrs import define, field
from emc.log import setup_logger
from emc.model.policy import Policy
from emc.data.constants import *
from collections import defaultdict
from math import isnan

logger = setup_logger(__name__)


@define
class Score:
    """
    Aggregates all objectives that can be used to score a policy on its quality
    """
    policy: Policy

    n_simulations: int

    n_wrong_classifications: int

    latenesses: list[int]

    sub_policy_costs: dict[Policy, dict[tuple[int, int], float]] = defaultdict(dict)

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

    def __float__(self):
        total_subpolicy_costs = [cost for sub_policy in self.policy.sub_policies for cost in
                                 self.sub_policy_costs[sub_policy].values() if not isnan(cost)]
        logger.info(
            f"- Totaal used simulations: {len(total_subpolicy_costs)} (nan: {self.n_simulations - len(total_subpolicy_costs)})")

        if len(total_subpolicy_costs):
            total_costs = sum(total_subpolicy_costs) / len(total_subpolicy_costs)
        else:
            total_costs = float('inf')
            logger.error("Found division by zero on line 324 of policy manager")
        logger.info(f"- Gemiddelde financiele kosten: {total_costs}")

        avg_lateness = sum(self.latenesses) / len(self.latenesses)
        penalty_costs = avg_lateness * RESISTANCE_NOT_FOUND_COSTS
        logger.info(
            f"- Total (avg) lateness: {sum(self.latenesses)} ({avg_lateness})")
        logger.info(f"- Gemiddelde penalty kosten: {penalty_costs}")

        total_costs += penalty_costs

        logger.info(f"-----------------\nTotal costs: {total_costs}")
        return float(total_costs)
