from emc.model import Scenario
from emc.model import Label


class LabelGenerator:
    def __init__(self, scenarios: list[Scenario]):
        self.scenarios = scenarios

    def generate(self):
        for scenario in self.scenarios:
            if (scenario.res_mode == 'none'):
                scenario.label_all_simulations(Label.NO_SIGNAL)
            else:
                scenario.label_all_simulations(Label.SIGNAL)
