from emc.data import DataLoader
from emc.data.infection_tree import InfectionTree
from emc.classifiers import GradientBoosting
from emc.data import LabelGenerator
from collections import Counter
from emc.util import data_path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def main():
    # df = pd.read_csv(data_path() / 'hookworm_monitor_age.csv')
    # df = df.drop(df.columns[0], axis=1)
    # df.to_csv('hookworm_monitor_age.csv', index=False)
    #
    # return
    worm = 'ascaris'
    loader = DataLoader(worm)
    scenarios = loader.load_scenarios()
    df = loader.monitor_age

    # for scen in range(16):
    #     start = 84 * (1000 * scen)
    #     df.loc[start:start + 1000, 'scen'] = scen + 1
    #
    #     for sim in range(1000):
    #         start = 84 * (1000 * scen + sim)
    #         df.loc[start:start + 84, 'sim'] = sim + 1

    df = df.sort_values(by=['scen', 'sim', 'time', 'age_cat']).reset_index()
    new_df = df.groupby(df.index // 4).agg(
        {'time': 'last', 'n_host': 'sum', 'n_host_eggpos': 'sum', 'a_epg_obs': 'sum', 'scen': 'last', 'sim': 'last'}
    ).reset_index(drop=True)

    def rate_of_change(col: pd.Series):
        new = col
        old = col.shift(1)

        return (new - old) / old

    new_df = new_df.sort_index()
    pred = new_df['n_host_eggpos'] > 0
    new_df.loc[pred, 'inf_level'] = new_df['n_host_eggpos'] / new_df['n_host']
    new_df.loc[~pred, 'inf_level'] = 0

    new_df['inf_level_change'] = rate_of_change(new_df['inf_level'])
    new_df['a_epg_obs_change'] = rate_of_change(new_df['a_epg_obs'])
    new_df.to_csv(f'{worm}_monitor_age_merged.csv', index=False)

    # NOTE: uncomment to generate new levels
    # Levels are currently saved in data, so you can instead retrieve them directly
    return
    tree = InfectionTree(scenarios, new_df)

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
