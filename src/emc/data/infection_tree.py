import pandas as pd
import numpy as np

from emc.model.scenario import Scenario
from emc.util import data_path


class InfectionTree:
    """
    Determines the infection levels for a given bucket size and plot them if available
    """

    def __init__(self, scenarios: list[Scenario], data: pd.DataFrame):

        self.scenarios = scenarios
        self.data = data
        self.buckets: dict[str, list[float]]

        self.__build()

    def __build(self):
        levels = np.arange(0, 1.0, 0.1)
        print(levels)

        level_simulations = {}

        for i, level in enumerate(levels):
            data = pd.DataFrame()

            print(f'({level}, {level + 0.1}]')

            for scenario in self.scenarios:
                if scenario.res_mode != 'none':
                    continue

                print(scenario.id)

                for simulation in scenario:
                    df = simulation.monitor_age
                    age_cats = df.groupby('age_cat')

                    for _, group in age_cats:
                        if level <= group.iloc[0]['inf_level'] < level + 0.1:
                            data = pd.concat([data, group])

            if data.empty:
                continue

            groups = data.groupby('time')
            mean_sd = []

            for time, group in groups:
                mean = group['inf_level'].mean()
                _max = group['inf_level'].max()
                _min = group['inf_level'].min()
                sd = group['inf_level'].std()
                mean_sd.append((mean, sd, _min, _max))

            level_simulations[int(10 * level)] = mean_sd

        with open(data_path() / 'levels.txt', 'w') as file:
            file.write(str(level_simulations))
