from pathlib import Path
from typing import Tuple, TypeVar, Optional
import pandas as pd
import json

# Type definitions
T = TypeVar('T')
Pair = Tuple[T, T]


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
        'drug_efficacy': 'csv',
        'levels': 'json',
        'metadata': 'json',
        'monitor_age': 'csv',
    }

    @classmethod
    def __safe_path(cls, path: Path) -> Path:
        """
        Get the safe version of a path, i.e. ensure all its parents exist
        :param path: Path to provide safety for
        :return: Same path, with as post-condition that all its parents exist
        """
        parent = path if path.is_dir() else path.parent
        parent.mkdir(parents=True, exist_ok=True)

        return path

    @classmethod
    def source(cls) -> Path:
        """
        Access the source path of the project from anywhere
        :return: Root folder of the project
        """
        return cls.__safe_path(cls.__ROOT / 'src' / 'emc')

    @classmethod
    def data(cls, typ: str = '.') -> Path:
        """
        Access the data path of the project from anywhere
        :return: Data folder of the project
        """
        return cls.__safe_path(cls.__ROOT / 'data' / typ)
    
    @classmethod
    def hyperparameter_opt(cls, filename : str) -> Path:
         """
        Access the hyperparameter path of the project from anywhere
        :return: hyperparameter folder of the project
        """
         return cls.__safe_path(cls.__ROOT / 'data' / 'hyperparameter' / filename)

    @classmethod
    def worm_data(cls, worm: str, data_type: str, use_merged: bool = True) -> Path:
        """
        Access the data path of the project given some parameters
        :param worm: Type of worm
        :param data_type: Type of data source
        :param use_merged: Whether to use the merged data source
        :return: Path to the data file
        """
        assert data_type != 'levels', "Use `level_data` instead and specify the mda freq and strategy"

        merge_str = '_merged' if use_merged and data_type == 'monitor_age' else ''
        ext = cls.__DATA_EXTENSIONS[data_type]

        path = cls.data(data_type) / f'{worm}{merge_str}.{ext}'
        return cls.__safe_path(path)

    @classmethod
    def levels(cls, worm: str, *, bucket_size: int, mda_freq: Optional[int], mda_strategy: Optional[str],
               baseline: Optional[int] = None):
        """
        Access the data path of the project given some parameters
        :param worm: Name of the worm
        :param bucket_size: Size of the buckets for the baseline infection level
        :param mda_freq: De-worming frequency, if relevant
        :param mda_strategy: De-worming population, if relevant
        :param baseline: Baseline level to get the path for if relevant
        :return: Name of the levels in the data
        """
        freq_str = f'{mda_freq}year' if mda_freq else 'any_freq'
        strategy_str = mda_strategy if mda_strategy else 'anyone'

        plot = baseline is not None
        ext = 'png' if plot else cls.__DATA_EXTENSIONS['levels']
        name = 'levels' + (str(baseline) if plot else '')

        folder = f'{freq_str}_{strategy_str}'
        file = f'{name}.{ext}'

        path = cls.data('levels') / worm / str(bucket_size) / folder / file
        return cls.__safe_path(path)

    @classmethod
    def stats(cls):
        return cls.__safe_path(cls.data('statistics') / 'stats.json')

class Writer:
    """
    Utility class to quickly export data
    """

    @classmethod
    def __read_json_file(cls, path):
        """
        Read JSON file from the given filename
        :param filename: Name of the JSON file
        :return: Data loaded from the JSON file
        """
        try:
            with open(path, 'r') as file:
                data = json.load(file)
            return data
        except FileNotFoundError:
            return {}
    
    @classmethod
    def __write_json_file(cls, path, data):
        """
        Write data to a JSON file
        :param filename: Name of the JSON file
        :param data: Data to be written
        """
        try:
            with open(path, 'w') as file:
                json.dump(data, file, indent=4)
        except Exception as e:
            print(f"Error writing to JSON file: {e}")
    
    @classmethod
    def update_json_file(cls, path, key, value):
        """
        Update JSON file with a key-value pair
        :param filename: Name of the JSON file
        :param key: Key to update
        :param value: Value to update
        """
        data = cls.__read_json_file(path)
        data[key] = value
        cls.__write_json_file(path, data)