import typing
from math import isnan
from typing import Optional

import pandas as pd
from attrs import define, field

# from emc.model.policy import Policy
from emc.log import setup_logger
from emc.util import Paths

# Required to avoid circular dependency
if typing.TYPE_CHECKING:
    from emc.model.scenario import Scenario

logger = setup_logger(__name__)


@define
class Simulation:
    """
    Combines all information about a single simulation for a single scenario
    Survey data is split into monitor_age (epidemiological) and drug_efficacy (drug efficacy) data frames
    """

    # Scenario the simulation belongs to
    scenario: "Scenario"

    # Frequency and timing of de-worming
    mda_time: list[int]

    # Targeted age group
    mda_age: range

    # Proportion of the target population that is effectively treated during each PC round
    mda_cov: float

    # Data concerning the simulated epidemiological survey results
    monitor_age: pd.DataFrame = field(eq=False, repr=False)

    # Data concerning the simulated drug efficacy survey results
    drug_efficacy_s: typing.Optional[pd.DataFrame] = field(eq=False, repr=False)

    # Identifier of the simulation
    id: int = field(eq=False, default=-1)

    def calculate_cost(self, policy: "Policy", allow_average: bool = True) -> float:
        """
        Calculate the cost of a given policy for the simulation
        :param policy: Policy to determine cost for
        :param allow_average: Whether cost is allowed to be calculated based on average over all policy years
        :return: Cost of the policy
        """
        costs = policy.calculate_cost(self.drug_efficacy_s, allow_average=allow_average)
        if isnan(costs):
            logger.debug(f"NaN costs for {self.scenario.id, self.id}")

        return costs

    def calculate_drug_cost(self, policy: "Policy", year: Optional[int] = None) -> float:
        """
        Calculate the drug survey cost of a given policy for the simulation
        :param policy: Policy to determine cost for
        :param year: Year to schedule if any, otherwise take an average over all years
        :return: Drug cost of the policy
        """
        costs = policy.calculate_drug_cost(self.drug_efficacy_s, year)
        if isnan(costs):
            logger.debug(f"NaN costs for {self.scenario.id, self.id}")

        return costs

    def verify(self, policy) -> Optional[float]:
        """
        Verify whether drug efficacy becomes a problem under the given policy
        :param policy: Policy to find drug efficacy from
        :return: Drug efficacy of the survey the year after the policy ends
        """
        df = self.drug_efficacy_s

        # Determine the year in which the drug efficacy survey would be scheduled
        # If it falls outside the simulated data, give no prediction
        year = policy.last_year + 1
        if year not in df['time'].values:
            return None

        # Otherwise, use the ERR for the given year as prediction if it is valid
        ERR = df.loc[df['time'] == year, 'ERR']
        if ERR.empty:
            return None

        return ERR.iloc[0]

    @property
    def unmerged_drug_efficacy_s(self):
        worm = self.scenario.species
        path = Paths.host_data(worm, self.scenario.id, self.id)

        return pd.read_csv(path)

    def __hash__(self):
        return hash((self.scenario.id, self.id))
