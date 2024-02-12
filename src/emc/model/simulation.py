import typing

import pandas as pd
from attrs import define

from emc.model.label import Label

# Required to avoid circular dependency
if typing.TYPE_CHECKING:
    from emc.model.scenario import Scenario

from emc.data.data_model import DataModel


@define
class Simulation:
    id: int

    scenario: "Scenario"

    # Frequency and timing of de-worming
    mda_time: list[int]

    # Targeted age group
    mda_age: range

    # Proportion of the target population that is effectively treated during each PC round
    mda_cov: float

    monitor_age: pd.DataFrame

    label: Label

    def filter_cond(self, df: pd.DataFrame):
        return (df['scen'] == self.scenario.id) & (df['sim'] == self.id)
