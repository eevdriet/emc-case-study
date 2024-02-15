import pandas as pd
from attrs import define

from emc.data.data_model import DataModel
from emc.model.label import Label
from emc.model.simulation import Simulation

from typing import Iterator


@define
class Scenario:
    """
    Combines all information about the simulations of a single scenario
    Survey data is split into monitor_age (epidemiological) and drug_efficacy (drug efficacy) data frames
    """
    id: int

    # Worm species ("hook", "asc")
    species: str

    # De-worming frequency per year
    mda_freq: int

    # Targeted population ("sac" for school-age children, "community" for the wider community)
    mda_strategy: str

    # Initial frequency of resistant alleles in the worm population
    res_freq: int

    # Mode of inheritance of resistance ("none", "recessive", "co-dominant", "dominant)
    res_mode: str

    # Information about the simulation for this specific scenario
    simulations: list[Simulation] = []

    def __getitem__(self, sim_idx: int):
        """
        Retrieve specific simulation for this scenario
        :param sim_idx: Index of the simulation
        :return: Specific simulation
        """
        assert 0 <= sim_idx < len(self.simulations)

        return self.simulations[sim_idx]

    def __iter__(self) -> Iterator:
        """
        Iterator method to iterate over simulations in the scenario
        :return: Iterator of the class
        """
        return iter(self.simulations)

    def __next__(self) -> Simulation:
        """
        Returns the next simulation in the scenario
        :return: The next simulation
        """
        return next(iter(self.simulations))

    def filter_cond(self, df: pd.DataFrame):
        return df['scen'] == self.id
