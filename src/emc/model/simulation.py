import typing

import pandas as pd
from attrs import define, field

from emc.model.label import Label
from emc.model.policy import Policy

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

    def calculate_cost(self, policy: Policy):
        return policy.calculate_cost(self.drug_efficacy_s)
