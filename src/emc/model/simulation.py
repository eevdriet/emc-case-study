from attrs import define

from emc.model import Scenario


@define
class Simulation:
    id: int
    scenario: Scenario
