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

    # step = 84 * 1000
    # for scen in range(16):
    #     start = step * scen
    #     end = start + step
    #     df.loc[start:end, 'scen'] = int(scen + 1)
    #
    # step = 84
    # for scen in range(16):
    #     for sim in range(1000):
    #         start = step * (1000 * scen + sim)
    #         end = start + step
    #         df.loc[start:end, 'sim'] = int(sim + 1)
    #
    # df.to_csv('ascaris.csv', index=False)

    # Infection tree
    # tree = InfectionTree(scenarios, loader.monitor_age)

    with open(data_path() / 'levels.txt', 'r') as file:
        level_simulations = eval(file.read())

    # level_simulations = 4
    # print(level_simulations)

    means, sds, mins, maxs = zip(*level_simulations[0.3])
    times = range(len(means))
    plt.errorbar(times, means, yerr=sds, color='b')
    # plt.plot(times, means, 'o-', color='b')
    plt.ylim(0, 1)
    # plt.grid(True)
    plt.show()

    # Classifiers bouwen
    # gb = GradientBoosting()
    # print(gb.run(scenarios))


if __name__ == "__main__":
    main()
