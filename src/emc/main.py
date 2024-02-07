from emc.model import Scenario, Simulation


def main():
    scenario = Scenario(1)
    simulation = Simulation(1, scenario)

    print(simulation.id)


if __name__ == "__main__":
    main()
