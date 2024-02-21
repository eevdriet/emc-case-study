import json
import scipy as sp
from emc.data.constants import *
from emc.util import Paths
from itertools import product
from collections import defaultdict
from typing import Tuple
from math import isnan


class WelchTest:
    """
    Performs Welch-T tests for all infection levels for each time point
    The result of a single test is as follows:
    Not rejected -> Means of infection level are equal between 'none' and the other resistance mode
    Rejected     -> Means of infection level are different between 'none' and the other resistance mode
    """

    __BUCKET_SIZE = 10
    Scenario = Tuple[str, str, int]  # worm, strategy, frequency

    def __init__(self):
        self.stats = self.__init_statistics()

    @classmethod
    def __init_statistics(cls) -> dict:
        """
        Initialize the statistics
        :return: Initial statistics
        """
        counts = {'not_rejected': 0, 'rejected': 0}
        return {
            'time': {str(time): counts.copy() for time in range(N_YEARS)},
            'baseline': {str(bucket): counts.copy() for bucket in range(0, 100, cls.__BUCKET_SIZE)},
            'strategy': {str(strategy): counts.copy() for strategy in MDA_STRATEGIES},
            'frequency': {str(frequency): counts.copy() for frequency in MDA_FREQUENCIES},
            'res_mode': {res_mode: counts.copy() for res_mode in set(RESISTANCE_MODES) - {"none"}},
        }

    def __collect_statistics(self, scenario: Scenario, levels: dict):
        """
        Collect the statistics for the levels of a given scenario
        :param scenario: Specific scenario to collect statistics for
        :param levels: Infection levels to collect statistics from
        """
        # Keep track of how many tests are (not) rejected for the current scenario
        scenario_stats = {'not_rejected': 0, 'rejected': 0}
        worm, strategy, frequency = tuple(map(str, scenario))

        def count(stats: dict, is_rejected: bool):
            if is_rejected:
                stats['rejected'] += 1
            else:
                stats['not_rejected'] += 1

        for baseline in levels:
            none_levels = levels[baseline]['none']

            for res_mode in levels[baseline]:
                # Only compare other resistance modes to 'none'
                if res_mode == 'none':
                    continue

                # Find levels to compare to
                comp_levels = levels[baseline][res_mode]

                for time, (none_level, comp_level) in enumerate(zip(none_levels, comp_levels)):
                    none_mean, none_sd, *rest, none_n = none_level
                    comp_mean, comp_sd, *rest, comp_n = comp_level

                    # Skip missing data
                    if isnan(none_mean) or isnan(comp_mean):
                        continue

                    # Perform T-test to verify whether the means are equal between 'none' and the other resistance mode
                    t_stat, p_val = sp.stats.ttest_ind_from_stats(none_mean, none_sd, none_n, comp_mean, comp_sd,
                                                                  comp_n)

                    # Update all relevant statistics
                    all_stats = (
                        scenario_stats,
                        self.stats['time'][str(time)], self.stats['baseline'][baseline],
                        self.stats['res_mode'][res_mode], self.stats['strategy'][strategy],
                        self.stats['frequency'][frequency]
                    )
                    for stats in all_stats:
                        count(stats, p_val < 0.05)

        self.stats[str(scenario)] = scenario_stats

    def test(self, save: bool = True):
        """
        Perform the Welch tests for all level data
        :param save: Whether to save the results to a JSON file
        :return: Dictionary of the test results
        """

        # Go through all scenarios
        for worm in Worm:
            worm = worm.value
            print(f"Collecting statistics for the {worm} worm...")

            for strategy, freq in product(MDA_STRATEGIES, MDA_FREQUENCIES):
                scenario = (worm, strategy, freq)
                print(f"\t- {scenario}")

                # Load the levels file
                path = Paths.levels(worm, bucket_size=self.__BUCKET_SIZE, mda_freq=freq, mda_strategy=strategy)
                with open(path, 'r') as file:
                    levels = json.load(file)

                # Update the statistics for the given levels
                self.__collect_statistics(scenario, levels)

        if save:
            path = Paths.stats()
            with open(path, 'w') as file:
                json.dump(self.stats, file, allow_nan=True, indent=4)


if __name__ == "__main__":
    test = WelchTest()
    test.test(save=True)
