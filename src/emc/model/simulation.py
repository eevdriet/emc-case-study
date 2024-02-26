import typing

import pandas as pd
from attrs import define, field
from typing import Optional
from math import isnan

from emc.model.label import Label
from emc.data.constants import *

# from emc.model.policy import Policy

# Required to avoid circular dependency
if typing.TYPE_CHECKING:
    from emc.model.scenario import Scenario


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

    # Label that defines what signal the simulation produces
    label: Label

    # Identifier of the simulation
    id: int = field(eq=False, default=-1)

    def calculate_cost(self, policy):
        """
        Calculate the cost of a given policy for the simulation
        :param policy: Policy to determine cost for
        :return: Cost of the policy
        """
        costs = policy.calculate_cost(self.drug_efficacy_s)
        if isnan(costs):
            print(f"ERROR: nan costs for {self.scenario.id, self.id}")

        return costs

    def predict(self, policy) -> Optional[float]:
        """
        Predict whether drug efficacy becomes a problem under the given policy
        :param policy: Policy to find drug efficacy from
        :return: Drug efficacy of the survey the year after the policy ends
        """
        df = self.drug_efficacy_s

        # Determine the year in which the drug efficacy survey would be scheduled
        # If it falls outside the simulated data, give no prediction
        year = policy.last_year + 1
        if year not in df['time'].values:
            return None

        # Otherwise, use the ERR for the given year as prediction
        return df.loc[df['time'] == year, 'ERR'].iloc[0]
