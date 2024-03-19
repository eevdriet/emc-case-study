import json
from collections import defaultdict
from math import isnan
from pathlib import Path

import matplotlib.pyplot as plt

from emc.data.constants import *
from emc.data.data_loader import DataLoader
from emc.log import setup_logger
from emc.model.policy import Policy
from emc.util import Paths

logger = setup_logger(__name__)

Costs = dict[int, list[float]]


def generate_costs(worm: str, path: Path, mda_freq: int, mda_strategy: str) -> Costs:
    loader = DataLoader(worm)
    scenarios = loader.load_scenarios()
    scenarios = [
        s for s in scenarios
        if s.mda_strategy == mda_strategy and s.mda_freq == mda_freq
    ]

    costs = defaultdict(dict)
    policy = Policy.from_timepoints([0])

    for scenario in scenarios:
        for simulation in scenario:
            key = (simulation.scenario.id, simulation.id)
            logger.info(key)
            avg_cost = simulation.calculate_drug_cost(policy)
            costs[str(key)]["0"] = avg_cost

            for year in range(1, N_YEARS):
                cost = simulation.calculate_drug_cost(policy, year)
                costs[str(key)][str(year)] = cost

    path.parent.mkdir(exist_ok=True, parents=True)
    with open(path, 'w') as file:
        json.dump(costs, file, allow_nan=True, indent=4)

    return costs


def plot_costs(costs: Costs, path: Path):
    all_data = [costs[key] for key in costs if key > 0]
    boxplot_data = [[cost for cost in time_costs if not isnan(cost)] for time_costs in all_data]
    nan_counts = [len(all_costs) - len(clean_costs) for all_costs, clean_costs in zip(all_data, boxplot_data)]

    plt.gcf().set_facecolor('none')

    # Boxplot for the costs per time point
    fig, ax1 = plt.subplots()
    ax1.set_xticks(range(1, N_YEARS))
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Drug efficacy survey cost")
    bp = ax1.boxplot(boxplot_data)
    for median in bp['medians']:
        median.set_color(BLUE)

    # Line displaying number of missing observations
    ax2 = ax1.twinx()
    line = ax2.plot(range(1, N_YEARS - 1), nan_counts[:-1], color=MAGENTA, linestyle='-', marker='o', label='Mean')
    ax2.set_ylabel("Number of missing data per time point")

    # Combined legend
    legend = fig.legend([bp['medians'][0], line[0]], ['Median survey cost', 'Missing data count'], ncols=2,
                        bbox_to_anchor=(0.5, -0.001), loc='upper center')
    legend.get_frame().set_alpha(0)

    # Write and discard
    plt.savefig(path, bbox_inches='tight', transparent=True)
    plt.clf()


def main():
    OVERWRITE = False

    for worm in Worm:
        worm = worm.value

        for mda_strategy in MDA_STRATEGIES:
            for mda_freq in MDA_FREQUENCIES:
                freq_str = f'{mda_freq}year' if mda_freq else 'any_freq'
                strategy_str = mda_strategy if mda_strategy else 'anyone'
                path = Paths.data('costs') / f"{worm}_{freq_str}_{strategy_str}.json"
                if not path.exists() or OVERWRITE:
                    costs = generate_costs(worm, path, mda_freq, mda_strategy)
                else:
                    with open(path, 'r') as file:
                        costs = json.load(file)

                merged_costs = {time: [vals[str(time)] for vals in costs.values()] for time in range(N_YEARS)}
                plot_costs(merged_costs, path.with_suffix('.png'))

                print("End")


if __name__ == '__main__':
    main()
