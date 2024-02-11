import math

import pandas as pd
from attrs import define


def cost():
    print("Reached")
    df = pd.read_excel(r"C:\Users\Marin\OneDrive - Erasmus University Rotterdam\Documents\Python\emc-case-study"
                       r"\data\Erasmus MC_cost data.xlsx")  # emc-case-study-data\Erasmus MC_cost data.xlsx
    categories = df.iloc[:, 0].dropna().drop([0, 10, 42])  # TODO: incorporate appropriate usage or just use paper
    print(categories)
    df = df['total cost per sample'].dropna()
    return consumable(df) + personnel(df) + transportation(df)


def consumable(df):
    N = 430  # TODO: Get the number of total hosts (all age categories together
    samples = 1
    aliquots = 1
    sampleCost = 0.57
    aliquotCost = 1.37  # TODO: Assume single KK for now, duplicate KK = 1.51
    return 2 * N * samples * (sampleCost + aliquots * aliquotCost)


def personnel(df):
    return 2 * days(df) * 4 * 22.50


def transportation(df):
    return days(df) * 90


def days(df):
    N = 430
    workers = 4  # Under assumption of single mobile field team: 1 nurse, three technicians
    timeAvailable = workers * 4 * 60 * 60  # In seconds
    c = 100  # TODO: Find average for egg count or adjust to simulation
    timeProcessing = N * (15 + 67 + 9) + pow(10, 2.3896 + 0.0661 * math.log10(pow(c + 1, 2)))
    return math.ceil(timeProcessing / timeAvailable)


def totalCost(deSurvey, epSurvey):
    total = 0
    deCost = cost()
    epCost = 1 / 2 * deCost

    for i in deSurvey:
        if i:
            total += deCost

    for i in epSurvey:
        if i:
            total += epCost
    return total


@define
class Policy:
    # At which moments in time to conduct a survey or not
    deSurvey: list[bool] = [False] * 21
    epSurvey: list[bool] = [True] * 21
    totalCost = totalCost(deSurvey, epSurvey)
    print("total cost: ", totalCost)
