from emc.data import DataLoader
from emc.data.infection_tree import InfectionTree
from emc.classifiers import GradientBoosting
from emc.data import LabelGenerator
from collections import Counter
from emc.util import data_path

import matplotlib.pyplot as plt


def main():
    loader = DataLoader('ascaris')
    scenarios = loader.load_scenarios()
    df = loader.monitor_age

    # NOTE: uncomment to generate new levels
    # Levels are currently saved in data, so you can instead retrieve them directly
    tree = InfectionTree(scenarios, loader.monitor_age)

    path = data_path() / 'levels.txt'
    if not path.exists():
        print("Levels does not exist, generate from InfectionTree")
        return

    with open(path, 'r') as file:
        level_simulations = eval(file.read())

    means, sds, mins, maxs = zip(*level_simulations[3])
    times = range(len(means))
    plt.errorbar(times, means, yerr=sds, color='b')
    plt.ylim(0, 1)
    plt.show()

    # Classifiers bouwen
    # gb = GradientBoosting()
    # print(gb.run(scenarios))


if __name__ == "__main__":
    main()
