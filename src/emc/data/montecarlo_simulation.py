import pandas as pd
import numpy as np
from scipy.stats import gamma
import simpy as simpy
import random as random
import statistics as statistics


class MCSimulation:
    # Parameters
    __SHAPE_SLIDE = 100
    __TOTAL_SIM = 1000

    def __init__(self, simulation: pd.DataFrame, worm: str):
        self.sim = simulation
        if worm == "ASCARIS":
            self.__SHAPE_DAY = 0.510

        if worm == "HOOKWORM":
            self.__SHAPE_DAY = 1.000

    def total_simulation(self):
        print("reached")
        result = self.sim.copy()
        for i in range(len(result)):
            if result.at[i, 'drug_efficacy.pre'] != 0 and not np.isnan(result.at[i, 'drug_efficacy.pre']):
                result.at[i, 'drug_efficacy.pre2'] = self.__simulate_count_ids(result.at[i, 'drug_efficacy.pre'])
            else:
                result.at[i, 'drug_efficacy.pre2'] = 0

            if result.at[i, 'drug_efficacy.post'] != 0 and not np.isnan(result.at[i, 'drug_efficacy.post']):
                result.at[i, 'drug_efficacy.post2'] = self.__simulate_count_ids(result.at[i, 'drug_efficacy.post'])
            else:
                result.at[i, 'drug_efficacy.post2'] = 0
        return result

    def __simulate_count_ids(self, mu_i: float):
        mu_ids = self.__simulate_mu_ids(mu_i) # mu_ids = self.__simulate_mu_ids(mu_i)
        count = 0
        for i in range(self.__TOTAL_SIM):
            count += np.random.poisson(mu_ids)
        print("count: ")
        print(count / self.__TOTAL_SIM)
        return count / self.__TOTAL_SIM

    def __simulate_mu_ids(self, mu_i: float):
        mu_id = self.__simulate_mu_id(mu_i)
        mu_ids = 0
        for i in range(self.__TOTAL_SIM):
            mu_ids += gamma.rvs(self.__SHAPE_SLIDE, self.__SHAPE_SLIDE / mu_id)
        print("mu_ids: ")
        print(mu_ids)
        print(mu_ids / self.__TOTAL_SIM)
        return mu_ids / self.__TOTAL_SIM

    def __simulate_mu_id(self, mu_i: float):
        mu_id = 0
        for i in range(self.__TOTAL_SIM):
            mu_id += gamma.rvs(self.__SHAPE_DAY, self.__SHAPE_DAY / mu_i)
        print("mu_id: ")
        print(mu_id / self.__TOTAL_SIM)
        return mu_id / self.__TOTAL_SIM

    def __simulate_mu_id_14(self, mu_i: float):
        total = 0
        for _ in range(self.__TOTAL_SIM):
            mu_id = mu_i
            for i in range(14):
                mu_id = gamma.rvs(self.__SHAPE_DAY, self.__SHAPE_DAY / mu_id)

            total += mu_id

        return total / self.__TOTAL_SIM


def main():
    # Import data
    df_sim = pd.read_csv("ascaris_drug_efficacySC01SIM0001.csv")
    print("STARTTT")
    mc_simulation = MCSimulation(df_sim, "ASCARIS")
    result = mc_simulation.total_simulation()
    result.to_csv("result.csv")

    print(result)


if __name__ == '__main__':
    main()
