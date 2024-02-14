from pathlib import Path
import pandas as pd


def main():
    """
    Label the CSV files with the right scenario/simulation numbers
    :return: Nothing, just update existing CSV file
    """

    # Whether to change the merged or normal CSV files
    merged = True
    worm = 'ascaris'

    merge_str = '_merged' if merged else ''
    step = 21 if merged else 84

    path = Path.cwd() / '..' / 'data' / f'{worm}_monitor_age{merge_str}.csv'
    df = pd.read_csv(path)

    # Potentially columns with the wrong name, remove them if so
    drop_cols = [col for col in ['sim', 'scen'] if col in df.columns]
    if drop_cols:
        df = df.drop(drop_cols, axis=1)

    # Start adding new columns for scenario and simulation
    for scenario in range(16):
        start = step * 1000 * scenario
        df.loc[start:start + step * 1000, 'scenario'] = scenario + 1

        for simulation in range(1000):
            start = step * (1000 * scenario + simulation)
            df.loc[start:start + step, 'simulation'] = simulation + 1

    df.to_csv(path, index=False)


if __name__ == '__main__':
    main()
