from pathlib import Path
import pandas as pd


def normalised(series: pd.Series, missing_val: float = 0.5):
    """
    Get the normalised version of a series
    :param series: Series to normalise
    :param missing_val: Value to fill out when normalisation is invalid
    :return:
    """
    min_val = series.min()
    max_val = series.max()

    # In case normalisation is impossible, set all values as missing
    if min_val == max_val:
        return pd.Series([missing_val] * len(series), index=series.index)

    # Otherwise, normalise between 0 and 1
    return (series - min_val) / (max_val - min_val)


class Paths:
    """
    Utility class to quickly navigate the folder structure of the project
    """
    __ROOT = Path(__file__).parent.parent.parent
    __DATA_EXTENSIONS = {
        'monitor_age': 'csv',
        'metadata': 'json',
        'drug_efficacy': 'csv'
    }

    @classmethod
    def source(cls) -> Path:
        """
        Utility function to access the source path of the project from anywhere
        :return: Root folder of the project
        """
        return cls.__ROOT / 'src' / 'emc'

    @classmethod
    def data(cls, typ: str = '.') -> Path:
        """
        Utility function to access the data path of the project from anywhere
        :return: Data folder of the project
        """
        return cls.__ROOT / 'data' / typ

    @classmethod
    def worm_data(cls, worm: str, data_type: str, use_merged: bool = True) -> Path:
        """
        Utility function to access the data path of the project for a
        - specific worm species
        - specific data source (monitor_age / drug_efficacy)
        - merged or normal data
        :param worm: Type of worm
        :param data_type: Type of data source
        :param use_merged: Whether to use the merged data source
        :return: Path to the data file
        """
        merge_str = '_merged' if use_merged and data_type == 'monitor_age' else ''
        ext = cls.__DATA_EXTENSIONS[data_type]

        return cls.data(data_type) / f'{worm}{merge_str}.{ext}'
