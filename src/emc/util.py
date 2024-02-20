from pathlib import Path


class Paths:
    """
    Utility class to quickly navigate the folder structure of the project
    """
    __ROOT = Path(__file__).parent.parent.parent

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
    def worm_data(cls, worm: str, typ: str, use_merged: bool = True) -> Path:
        """
        Utility function to access the data path of the project for a
        - specific worm species
        - specific data source (monitor_age / drug_efficacy)
        - merged or normal data
        :param worm: Type of worm
        :param typ: Type of data source
        :param use_merged: Whether to use the merged data source
        :return: Path to the data file
        """
        merge_str = '_merged' if use_merged and typ == 'monitor_age' else ''
        ext = {
            'monitor_age': 'csv',
            'metadata': 'json',
            'drug_efficacy': 'csv'
        }[typ]

        return cls.data(typ) / f'{worm}{merge_str}.{ext}'
