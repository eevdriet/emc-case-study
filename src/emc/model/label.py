from enum import Enum


class Label(Enum):
    """
    Defines the result of a single simulation, i.e. whether a signal of drug resistance is detected
    """
    NO_SIGNAL = 0
    SIGNAL = 1
