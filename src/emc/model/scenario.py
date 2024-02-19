from typing import Iterator

from attrs import define, field

from emc.model.simulation import Simulation


@define
class Scenario:
    """
    Combines all information about the simulations of a single scenario
    Survey data is split into monitor_age (epidemiological) and drug_efficacy (drug efficacy) data frames
    Note that scenarios are only compared on their pre-determined settings, such as target population
    """

    # Worm species ("hook", "asc")
    species: str

    # De-worming frequency per year (annually: 1, bi-annually: 2)
    mda_freq: int

    # Targeted population ("sac" for school-age children, "community" for the wider community)
    mda_strategy: str

    # Initial frequency of resistant alleles in the worm population (between 0 and 1)
    res_freq: float = field(eq=False)

    # Mode of inheritance of resistance ("none", "recessive", "co-dominant", "dominant")
    res_mode: str = field(eq=False)

    # Identifier of the scenario
    id: int = field(eq=False, default=-1)

    # Information about the simulation for this specific scenario
    simulations: list[Simulation] = field(default=list(), eq=False, repr=False)

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
