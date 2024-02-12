from emc.data import DataLoader
from emc.classifiers import GradientBoosting
from emc.data import LabelGenerator
from collections import Counter


def main():
    loader = DataLoader('hookworm')
    scenarios = loader.load_scenarios()

    # Classifiers bouwen
    gb = GradientBoosting()


if __name__ == "__main__":
    main()
