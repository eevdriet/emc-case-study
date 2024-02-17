from emc.data import DataLoader
from emc.data.level_builder import LevelBuilder
from emc.classifiers import GradientBoosting
from emc.data import LabelGenerator
from emc.model.scenario import Scenario
from collections import Counter
from emc.util import data_path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def main():
    worm = 'ascaris'
    loader = DataLoader(worm, use_merged=False, load_efficacy=False)
    df = loader.monitor_age
    scenarios = loader.load_scenarios()


if __name__ == "__main__":
    main()
