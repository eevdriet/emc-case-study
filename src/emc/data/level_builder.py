import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from emc.model.scenario import Scenario
from emc.util import data_path
from emc.data.constants import *

Level = tuple[float, float, float, float, int]  # mean, sd, min, max, no. observations
Levels = dict[str, list[Level]]
ModeLevels = dict[str, Levels]


class LevelBuilder:
    """
    Determines the infection levels for a given bucket size and plot them if available
    """
    __RES_MODES = ["none", "dominant", "codominant", "recessive"]
    # __COLORS = ["red", "green", "blue", "yellow"]
    __COLORS = ["blue", "yellow", "green", "pink"]

    def __init__(self, scenarios: list[Scenario]):
        # Data
        self.all_scenarios = scenarios
        self.scenarios = []

        # Parameters
        self.bucket_size: float = 0.1
        self.mda_freq: Optional[int] = None
        self.mda_strategy: Optional[str] = None
        self.buckets: np.arange = np.arange(0, 1.0, self.bucket_size)

        # Output
        self.mode_levels: ModeLevels = defaultdict(dict)

    def build(self, bucket_size: int, *, mda_freq: int = None, mda_strategy: str = None,
              overwrite: bool = False) -> ModeLevels:
        """
        Build the infection levels for a given scenario
        :param bucket_size: Size of the buckets to classify the starting infection levels by
        :param mda_freq: De-worming frequency, if relevant
        :param mda_strategy: De-worming population, if relevant
        :param overwrite: Whether to overwrite saved infection levels data if existing
        :return: Infection levels for the scenario
        """
        self.bucket_size = bucket_size
        self.mda_freq = mda_freq
        self.mda_strategy = mda_strategy

        # Set up relevant buckets and scenarios
        self.buckets = range(0, 100, bucket_size)
        self.scenarios = self.__find_scenarios(mda_freq, mda_strategy)

        # Verify whether the levels were previously generated and should be overwritten
        name = self.__construct_name()
        path = data_path() / 'levels' / f'{name}.json'

        if path.exists() and not overwrite:
            with open(path, 'r') as file:
                self.mode_levels = json.load(file)
                return self.mode_levels

        # If not, build the levels and save them as JSON data
        for baseline in self.buckets:
            self.__build_levels(baseline)

        with open(path, 'w') as file:
            json.dump(self.mode_levels, file, allow_nan=True, indent=4)

        return self.mode_levels

    def plot(self, baseline: int, save: bool = False, show: bool = True):
        """
        Plot the most recently created infection levels
        :param baseline: Baseline infection level for which to plot
        :param save: Whether to save the plot
        :param show: Whether to show the plot
        """
        if not self.mode_levels:
            print("Levels do not exist, build first using `build_levels`")
            return

        levels = self.mode_levels[str(baseline)]
        times = range(19)

        for res_mode, color in zip(self.__RES_MODES, self.__COLORS):
            means, sds, mins, maxs, n_hosts = map(np.array, zip(*levels[res_mode]))
            times = range(len(means))

            # Plot infection level means
            # OPTION 1: shaded overlapping regions
            plt.errorbar(times, means, yerr=sds, label=res_mode, color=color)
            plt.fill_between(times, means - sds, means + sds, alpha=0.5)

            # Plot error levels for the infection level means
            # OPTION 2: dashed lines
            # plt.plot(times, means, '-o', color=color, label=res_mode)
            # for time, mean, sd in zip(times, means, sds):
            #     plt.vlines(x=time, ymin=mean - sd, ymax=mean + sd, color=color, linestyle='dotted')
            plt.xticks(range(21))

        # Titles
        worm = self.all_scenarios[0].species
        plt.title(f"Infection levels for the {worm} worm with a [{baseline}%, {baseline + self.bucket_size}%) baseline")

        freq = f"{self.mda_freq}x PC/year" if self.mda_freq is not None else "Any PC strategy"
        strat = self.mda_strategy if self.mda_strategy is not None else "Any target population"
        plt.text(0.05, 0.92, f'{freq}\n{strat}', horizontalalignment='left', transform=plt.gca().transAxes)

        # Labels
        plt.xlabel("Time (years)")
        plt.ylabel("Infection level")
        plt.ylim(0, 1)
        plt.legend(title="Resistance mode")

        if show:
            plt.show()
        if save:
            name = self.__construct_name()
            path = data_path() / 'levels' / 'figures' / f'{name}_{baseline}%'
            plt.savefig(path)

        plt.clf()

    def __build_levels(self, baseline: int):
        """
        Build the infection levels for a given baseline infection level
        :param baseline: Baseline infection level
        :return: Infection levels
        """

        print(f'[{baseline}, {baseline + self.bucket_size})')

        for res_mode in self.__RES_MODES:
            data = pd.DataFrame()

            for scenario in self.all_scenarios:
                if scenario.res_mode != res_mode:
                    continue
                if self.mda_freq is not None and scenario.mda_freq != self.mda_freq:
                    continue
                if self.mda_strategy is not None and scenario.mda_strategy != self.mda_strategy:
                    continue

                print(scenario.id)

                for simulation in scenario:
                    df = simulation.monitor_age

                    # Only include simulations that start at the requested baseline infection level
                    if baseline <= 100 * df.iloc[0]['inf_level'] < baseline + self.bucket_size:
                        data = pd.concat([data, df])

            if data.empty:
                continue

            # Find results per time point for all relevant simulations
            groups = data.groupby('time')
            bucket_levels = []

            for time, group in groups:
                level = (
                    group['inf_level'].mean(),
                    group['inf_level'].std(),
                    group['inf_level'].min(),
                    group['inf_level'].max(),
                    len(group['inf_level'])
                )
                bucket_levels.append(level)

            self.mode_levels[str(baseline)][res_mode] = bucket_levels

    def __find_scenarios(self, mda_freq: Optional[int], mda_strategy: Optional[str]) -> list[Scenario]:
        """
        Only include scenarios in level generation that are relevant to the current MDA policy
        :param mda_freq: De-worming frequency, if relevant
        :param mda_strategy: De-worming population, if relevant
        :return: Relevant scenarios
        """
        scenarios = []

        for scenario in self.all_scenarios:
            scenarios.append(scenario)

        return scenarios

    def __construct_name(self) -> str:
        """
        Construct the path of the data of a certain scenario
        :return: Path to the level data
        """
        worm = self.all_scenarios[0].species
        freq_str = f'{self.mda_freq}year' if self.mda_freq else 'any_freq'
        strat_str = self.mda_strategy if self.mda_strategy else 'anyone'

        return f'{worm}_{self.bucket_size}_{freq_str}_{strat_str}'


def build_levels(overwrite):
    from emc.data.data_loader import DataLoader
    from itertools import product

    # Load in the data
    for worm in Worm:
        worm = worm.value
        loader = DataLoader(worm, use_merged=True, load_efficacy=False)
        scenarios = loader.load_scenarios()

        # Set up a level builder and build all possible levels
        builder = LevelBuilder(scenarios)

        for bucket_size in [5, 10, 20]:
            mda_strategy = [None, 'sac', 'community']
            mda_freq = [None, 1, 2]

            for strat, freq in product(mda_strategy, mda_freq):
                print(f"-- {bucket_size} with {freq=}, {strat=}")
                builder.build(bucket_size, mda_strategy=strat, mda_freq=freq, overwrite=overwrite)

                for baseline in range(0, 70, bucket_size):
                    builder.plot(baseline, save=True, show=False)

    print("Done")


if __name__ == "__main__":
    build_levels(overwrite=False)
