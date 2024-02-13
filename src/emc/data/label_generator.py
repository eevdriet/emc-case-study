import pandas as pd
import numpy as np
from math import ceil, floor

from emc.model import Scenario, Simulation
from emc.util import data_path
from emc.model import Label


class LabelGenerator:
    def __init__(self, scenarios: list[Scenario]):
        self.scenarios = scenarios

    def generate(self):
        df = pd.DataFrame()

        for scenario in self.scenarios:
            print(f"Scenario {scenario.id}")

            for simulation in scenario.simulations:
                print(f"\t - {simulation.id:04}/1000")
                df = pd.concat([df, self.__preprocess(simulation)])

        path = data_path() / '..' / 'csv2' / f'{self.scenarios[0].species}_monitor_age.csv'
        df.to_csv(path)

    @classmethod
    def __preprocess(cls, simulation: Simulation) -> None:
        df = simulation.monitor_age

        # Calculate rate of change based on data frame column
        def rate_of_change(col: pd.Series):
            new = col
            old = col.shift(1)

            if old == 0 or new - old == 0:
                return np.nan
            else:
                return (new - old) / old

        def age_cat_func(group):
            age_cat = int(group['age_cat'].iloc[-1])

            # Add interesting columns
            pred = group['n_host_eggpos'] > 0
            group.loc[pred, 'inf_level'] = group['n_host_eggpos'] / group['n_host']
            group.loc[~pred, 'inf_level'] = 0

            group['inf_level_change'] = rate_of_change(group['inf_level'])
            group['a_epg_obs_change'] = rate_of_change(group['a_epg_obs'])

            # Simulation does not go fully to 20 years, add missing data
            last_year = group['time'].iloc[-1]
            if last_year != 20:
                last_year_c = ceil(last_year)
                if last_year != last_year_c and floor(last_year) != group['time'].iloc[-2]:
                    last_year_c -= 1

                group['time'].iloc[-1] = last_year_c

                # Add missing rows
                for year in range(last_year_c + 1, 21):
                    row = {'age_cat': [age_cat], 'time': [year], 'n_host': ['NaN'], 'n_host_eggpos': ['NaN'],
                           'a_epg_obs': ['NaN']}
                    group = pd.concat([group, pd.DataFrame(row)])

            return group

        df = df.groupby('age_cat').apply(age_cat_func)
        
        assert len(df) == 84

        return df
