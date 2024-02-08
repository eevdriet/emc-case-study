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
