import typing

import pandas as pd
from attrs import define

# Required to avoid circular dependency
if typing.TYPE_CHECKING:
    from emc.model import Scenario

from emc.data import DataModel


@define
class Simulation(DataModel):
    id: int

    scenario: "Scenario"

    # Frequency and timing of de-worming
    mda_time: list[int]

    # Targeted age group
    mda_age: range

    # Proportion of the target population that is effectively treated during each PC round
    mda_cov: float

    def filter_cond(self, df: pd.DataFrame):
        return (df['scen'] == self.scenario.id) & (df['sim'] == self.id)
