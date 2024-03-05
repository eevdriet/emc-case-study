import logging
from pathlib import Path
import pickle
from typing import Tuple, TypeVar, Optional, Any
import numpy as np
import pandas as pd
import json

from emc.log import setup_logger

# Type definitions
T = TypeVar('T')
Pair = Tuple[T, T]

# Logger setup
logger = setup_logger(__name__)


def first_or_mean(df: pd.DataFrame, col: str, year: Optional[Any]) -> Any:
    """
    Find the first occurrence of a given value of the series or its mean if no value is given
    :param df: Series to collect data from
    :param year: Value to find if relevant
    :return: Mean or first in the series
    """
    logger = logging.getLogger(__name__)

    if col not in df.columns:
        logger.warning(f"Column '{col}' not found when getting first/mean")
        return np.nan

    if year is None:
        return df[col].mean(skipna=True)

    if 'time' in df.columns:
        return df.loc[df['time'] == year, col].iloc[0]

    logger.warning(f"Column 'time' not found when getting first/mean")
    return np.nan


def normalised(series: pd.Series, missing_val: float = 0.5):
    """
    Get the normalised version of a series
    :param series: Series to normalise
    :param missing_val: Value to fill out when normalisation is invalid
    :return:
    """
    logger = logging.getLogger(__name__)

    min_val = series.min()
    max_val = series.max()

    # In case normalisation is impossible, set all values as missing
    if min_val == max_val:
        logger.warning(f"Same min/max value, using {missing_val} as default")
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
    def hyperparameter_opt(cls, filename : str, plotdata: str = False) -> Path:
         """
        Access the hyperparameter path of the project from anywhere
        :return: hyperparameter folder of the project
        """
         if plotdata:
             return cls.__safe_path(cls.__ROOT / 'data' / 'hyperparameter' / 'plotdata' / filename)
         else:
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
        """
        Access the statistics path of the project.

        :return: Path to the statistics directory
        """
        path = cls.data('statistics') / 'stats.json'
        return cls.__safe_path(path)

    @classmethod
    def log(cls):
        """
        Access the log path of the project.

        :return: Path to the log directory
        """
        path = cls.__ROOT / 'log'
        return cls.__safe_path(path)

    @classmethod
    def models(cls, worm: str, mda_freq: int, mda_strategy: str, constructor: str, filename: str) -> Path:
        """
        Access the model path of the project given parameters.

        :param worm: Name of the worm
        :param mda_freq: De-worming frequency
        :param mda_strategy: De-worming strategy
        :param filename: Name of the file
        :return: Path to the model file
        """
        path = cls.data('model') / str(constructor) / str(worm) / str(mda_strategy) / str(mda_freq) / str(filename)
        return cls.__safe_path(path)

    @classmethod
    def preprocessing(cls, worm: str, mda_freq: int, mda_strategy: str, filename: str) -> Path:
        """
        Access the model path of the project given parameters.

        :param worm: Name of the worm
        :param mda_freq: De-worming frequency
        :param mda_strategy: De-worming strategy
        :param filename: Name of the file
        :return: Path to the model file
        """
        path = cls.data('preprocessing') / str(worm) / str(mda_strategy) / str(mda_freq) / str(filename)
        return cls.__safe_path(path)


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
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, 'r') as file:
                data = json.load(file)
            return data
        except:
            return {}

    @classmethod
    def __write_json_file(cls, path: Path, data: Any):
        """
        Write data to a JSON file
        :param path: Path to the JSON file
        :param data: Data to be written
        """
        try:
            with open(path, 'w') as file:
                json.dump(data, file, indent=4)
        except Exception as e:
            logger.error(f"Error writing to JSON file: {e}")

    @classmethod
    def update_json_file(cls, path: Path, key: Any, value: Any):
        """
        Update JSON file with a key-value pair
        :param path: Path to the JSON file
        :param key: Key to update
        :param value: Value to update
        """
        data = cls.__read_json_file(path)
        data[key] = value
        cls.__write_json_file(path, data)

    @classmethod
    def get_value_from_json(cls, path, key):
        """
        Try to get a value from a
        :param path: Path to the JSON file
        :param key: Key for the value to find
        :return: Value corresponding to the key if it exists
        """
        try:
            with open(path, 'r') as file:
                data = json.load(file)
                if key in data:
                    return data[key]
                else:
                    return False
        except FileNotFoundError:
            logger.error(f"File not found: {path}")
            return False
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON format in the file: {path}")
            return False

    @classmethod
    def savePickle(cls, path: Path, data) -> None:
        """
        Save the model to a file using pickle serialization.

        :param path: Path to save the model file
        :param data: Data to save to the path
        :return: None
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as file:
            pickle.dump(data, file)

    @classmethod
    def loadPickle(cls, path: Path):
        """
        Load the data from a file.

        :param path: Path to the data file
        :return: Loaded data if the file exists, False otherwise
        """
        if path.exists():
            with open(path, 'rb') as file:
                data = pickle.load(file)
            return data
        else:
            return None

    @classmethod
    def saveNumpy(cls, path: Path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, data)

    @classmethod
    def loadNumpy(cls, path: Path):
        if path.exists():
            return np.load(path)
        else:
            return None


if __name__ == '__main__':
    print(normalised(pd.Series([248.91, 282.16, 180.80, 211.54])))
