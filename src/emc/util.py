from pathlib import Path


def source_path():
    """
    Utility function to access the source path of the project from anywhere
    :return: Root folder of the project
    """
    return Path(__file__).parent


def data_path():
    """
    Utility function to access the data path of the project from anywhere
    :return: Data folder of the project
    """
    return source_path().parent.parent / 'data'


def worm_path(worm: str, typ: str, use_merged: bool = False) -> Path:
    """
    Utility function to access the path of the data for a
    - specific worm species
    - specific data source (monitor_age / drug_efficacy)
    - merged or normal data
    :param worm: Type of worm
    :param typ: Type of data source
    :param use_merged: Whether to use the merged data source
    :return: Path to the data file
    """
    merge_str = '_merged' if use_merged else ''
    ext = {
        'monitor_age': 'csv',
        'metadata': 'json',
        'drug_efficacy': 'feather'
    }[typ]

    return data_path() / f'{worm}_{typ}{merge_str}.{ext}'
