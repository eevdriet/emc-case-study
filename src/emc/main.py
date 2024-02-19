from emc.data import DataLoader
from emc.data.constants import *


def main():
    worm = Worm.ASCARIS
    worm = worm.value
    loader = DataLoader(worm)
    df = loader.monitor_age

    scenarios = loader.load_scenarios()
    return 0


if __name__ == "__main__":
    main()
