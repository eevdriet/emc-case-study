import typing

import pandas as pd
from attrs import define

from emc.model.label import Label

# Required to avoid circular dependency
if typing.TYPE_CHECKING:
    from emc.model.scenario import Scenario

from typing import Optional


@define
class Simulation:
    """
    Combines all information about a single simulation for a single scenario
    Survey data is split into monitor_age (epidemiological) and drug_efficacy (drug efficacy) data frames
    """

    # Identifier of the simulation
    id: int

    # Scenario the simulation belongs to
    scenario: "Scenario"

    # Frequency and timing of de-worming
    mda_time: list[int]

    # Targeted age group
    mda_age: range

    # Proportion of the target population that is effectively treated during each PC round
    mda_cov: float

    # Data concerning the simulated epidemiological survey results
    monitor_age: pd.DataFrame

    # Data concerning the simulated drug efficacy survey results
    drug_efficacy_s: Optional[pd.DataFrame]

    # Label that defines what signal the simulation produces
    label: Label

    def filter_cond(self, df: pd.DataFrame):
        return (df['scen'] == self.scenario.id) & (df['sim'] == self.id)
