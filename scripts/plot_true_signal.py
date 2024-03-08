from emc.data.data_loader import DataLoader
from emc.log import setup_logger
from emc.data.constants import *
from emc.util import Paths
import random

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from math import isnan
from pathlib import Path

logger = setup_logger(__name__)

Signals = dict[tuple[int, int], int]
__TRAIN_TEST_SPLIT_SIZE: float = 0.2

NO_DATA = -1
NO_SIGNAL = -2
USING_TEST = False


def plot_signals(counter: Counter, path: Path):
    # Record how many simulations did not end with a true signal
    n_signals = sum(counter.values())
    n_no_data = counter[NO_DATA]
    n_no_signal = counter[NO_SIGNAL]
    # del counter[NO_DATA]
    # del counter[NO_SIGNAL]

    # Create histogram from the signals
    data = list(elem for elem in counter.elements() if elem >= 0)
    bins = np.arange(NO_SIGNAL, N_YEARS, 1)

    plt.gcf().set_facecolor('none')
    plt.hist(data, bins=bins, label='True signal (year)', color=BLUE)
    plt.hist([NO_DATA] * counter[NO_DATA], bins=bins, color=MAGENTA, label='Insufficient data (ID)')
    plt.hist([NO_SIGNAL] * counter[NO_SIGNAL], bins=bins, color=YELLOW, label='No signal (NS)')

    # Labels
    plt.xticks(bins + 0.5, ['NS', 'ID'] + list(bins[2:]))
    plt.xlim(NO_SIGNAL - 0.5, N_YEARS - 1)
    plt.xlabel("Time (years)")
    plt.ylabel("Number of signals found")
    legend = plt.legend(title="Type of signal", loc='upper center',
                        ncols=3,
                        edgecolor='black',
                        bbox_to_anchor=(0.5, -0.15),
                        frameon=True)  # , bbox_to_anchor=(0.5, -0.15))
    legend.get_frame().set_alpha(None)
    legend.get_frame().set_facecolor((1, 1, 1, 0))

    # Save and discard
    plt.savefig(path, bbox_inches='tight', transparent=True)
    plt.clf()


def main():
    for worm in Worm:
        worm = worm.value

        loader = DataLoader(worm)
        all_scenarios = loader.load_scenarios()

        for mda_strategy in MDA_STRATEGIES:
            for mda_freq in MDA_FREQUENCIES:
                # res_mode = 'codominant'

                counter = Counter()

                freq_str = f'{mda_freq}year' if mda_freq else 'any_freq'
                strategy_str = mda_strategy if mda_strategy else 'anyone'
                path = Paths.data('signal') / f"{worm}_{freq_str}_{strategy_str}{'_TEST' if USING_TEST else ''}.png"

                # Use the policy manager
                logger.info(f"-- {worm}: {mda_strategy} with {mda_freq} --")

                scenarios = [
                    s for s in all_scenarios
                    if s.mda_strategy == mda_strategy and s.mda_freq == mda_freq
                ]
                simulations = [simulation for scenario in scenarios for simulation in scenario]

                # Randomly order the simulations and split into train/validation
                if USING_TEST:
                    random.seed(SEED)
                    random.shuffle(simulations)

                    # Split the simulations
                    split_idx = int(len(simulations) * __TRAIN_TEST_SPLIT_SIZE)
                    train_sims = simulations[split_idx:]
                    test_sims = simulations[:split_idx]

                for simulation in (test_sims if USING_TEST else simulations):
                    print(simulation.scenario.id, simulation.id)

                    key = (simulation.scenario.id, simulation.id)
                    epi_signals = simulation.monitor_age['target']
                    drug_signals = simulation.drug_efficacy_s['ERR'][::simulation.scenario.mda_freq]

                    for year, (epi_signal, drug_signal) in enumerate(zip(epi_signals, drug_signals)):
                        # Cannot find signal from data
                        if isnan(epi_signal) or isnan(drug_signal):
                            counter[NO_DATA] += 1
                            break

                        if epi_signal < DRUG_EFFICACY_THRESHOLD and drug_signal < 0.85:
                            counter[year] += 1
                            break
                    else:
                        counter[NO_SIGNAL] += 1

                plot_signals(counter, path)


if __name__ == '__main__':
    main()
