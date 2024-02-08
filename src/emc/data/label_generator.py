from emc.model import Scenario
from emc.model import Label


class LabelGenerator:
    def __init__(self, scenarios: list[Scenario]):
        self.scenarios = scenarios

    def generate(self):
        df = self.scenarios[0]._epi_data
        no_resistance = {scenario.id for scenario in self.scenarios if scenario.res_mode == 'none'}
        df.loc[df['scen'].isin(no_resistance), 'label'] = 0  # Label.NO_SIGNAL
        df.loc[~df['scen'].isin(no_resistance), 'label'] = 1  # Label.SIGNAL
