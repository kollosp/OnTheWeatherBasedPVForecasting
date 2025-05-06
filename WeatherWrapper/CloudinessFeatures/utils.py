import numpy as np
# from collections.abc import Iterable
from typing import Callable, Iterable


def apply_window_function(func: Callable[[Iterable[float]], Iterable[float]], data: Iterable[float], window_size: int,
                          roll: bool = True) -> Iterable[float]:
    """
    Function performs the given function on windows
    :param func: function to be used f(d:np.ndarray) -> [float | int]
    :param data: array to be calculated
    :param window_size: size of the window
    :param roll: if True, then data are shifted after operation
    :return: array that contains the results of func for each window
    """
    d = np.zeros(data.shape)
    d[window_size:] = np.array([func(data[i - +window_size:i]) for i in range(window_size, len(data))])

    if roll:
        d = np.roll(d, -int(window_size / 2))

    return d

def window_subtraction(data1: np.ndarray, data2: np.ndarray, window_size: int, roll: bool = True) -> np.ndarray:
    """ Function subtruct one timeseries from another """

    d = np.zeros(data1.shape)
    d[window_size:] = np.array(
        [np.sum(data1[i - window_size:i] - data2[i - window_size:i]) for i in range(window_size, len(data1))])

    if roll:
        d = np.roll(d, -int(window_size / 2))

    return d

def window_moving_avg(data: np.ndarray, window_size: int, roll: bool = True) -> np.ndarray:
    """ Function compute moving average using selected window """
    return apply_window_function(np.mean, data, window_size, roll)

def max_pool(data: Iterable[float], window_size: int, roll: bool = True):
    """Take max from timeseries over a specified window"""
    return apply_window_function(np.max, data, window_size, roll)

def min_pool(data: Iterable[float], window_size: int, roll: bool = True):
    """Take min from timeseries over a specified window"""
    return apply_window_function(np.min, data, window_size, roll)