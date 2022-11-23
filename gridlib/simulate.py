"""
Module with function to sample from a provided GRID spectrum or multi-exponential.
"""
from typing import Sequence, Union
import math

import numpy as np

from . import data_utils


def tl_simulation_single(
    k: np.ndarray,
    s: np.ndarray,
    kb: float,
    t_int: float,
    t_tl: float,
    N: int = 10000,
):
    """Function simulates the dissociation of molecules and calculates the resulting
    survival time distribution.

    Parameters
    ----------
    k: np.ndarray
        Decay rates with units per second.
    s: np.ndarray
        Amplitudes for the respective decay rates.
    kb: float
        Photobleaching rate per second (kb = a / t_int).
    t_int: float
        The integration time in seconds.
    t_tl: float
        The time-lapse time to simulate in seconds.
    N: int, optional
        Number of molecules to simulate and use for the survival time distribution
        calculation (default 10000).

    Returns
    -------
    Dict[str, Dict[str, np.ndarray]]
        Survival function data for specified time-lapse condition with the following
        data structure:
            {
                "t_tl": {
                    "time": np.ndarray with the time points,
                    "value": np.ndarray with the survival function values,
                }
            }
    """

    if not math.isclose(1.0, np.sum(s)):
        raise ValueError(f"Sum of amplitudes should be equal to 1, but is {np.sum(s)}.")

    if kb <= 0:
        raise ValueError("'kb' argument should be larger than zero")
    if t_int <= 0:
        raise ValueError("'t_int' argument should be larger than zero")
    if t_tl <= 0:
        raise ValueError("'t_tl' argument should be larger than zero")
    if N <= 0:
        raise ValueError("'N' argument should be larger than zero.")

    p = s / np.sum(s)

    # Can also change kb * t_int to a
    keff = (kb * t_int) / t_tl + k

    count = 0
    time = np.linspace(t_tl, 1000 * t_tl, num=1000, endpoint=True)

    binding = np.zeros(time.shape[0])
    binding_sum = 0

    # TODO: future - improve speed: multiprocessing, or use "chunks", so do like
    # 1000 molecules at the same time with arrays?
    rng = np.random.default_rng()

    # Simulate until enough points have been captured or the computation time is
    # exceeded
    while binding_sum < N and count < (N * 10):

        keff_state = rng.choice(keff, p=p)
        lifetime = rng.exponential(scale=1 / keff_state)

        if lifetime > time[0]:
            idx = np.sum(lifetime > time) - 1  # - 1 because indexing starts at 0
            binding[idx] = binding[idx] + 1
            binding_sum = binding_sum + 1
        else:
            count = count + 1

    # Calculate the survival function values with cumulative summing
    value = np.cumsum(binding[::-1])[::-1]

    # Delete the empty elements
    time = time[value > 0]
    value = value[value > 0]

    data = {f"{t_tl}s": {"time": time, "value": value}}

    # Make sure the data is formatted correctly
    data = data_utils.fmt_t_str_data(data)

    return data


def tl_simulation(
    k: np.ndarray,
    s: np.ndarray,
    kb: float,
    t_int: float,
    t_tl_all: Sequence[Union[float, str]],
    N: Union[int, Sequence[int]] = 10000,
):
    """Function simulates the dissociation of molecules and calculates the resulting
    survival time distribution.

    Parameters
    ----------
    k : np.ndarray
        Decay rates with units per second.
    s : np.ndarray
        Amplitudes for the respective decay rates.
    kb : float
        Photobleaching rate per second (kb = a / t_int).
    t_int : float
        The integration time in seconds.
    t_tl_all : Sequence[float | str]
        The time-lapse times to simulate. If the time-lapse times are provided as
        floats then they are interpreted as seconds. However, the time-lapse times can
        also be provided as strings, eg. '100ms' or '1.5s'.
    N : int | Sequence[int], optional
        Number of molecules to simulate and use for the survival time distribution
        calculation (default 10000). If only an integer value is provided then this
        value will be used for all the simulations. In the case that the provided
        N object is an iterable, each element is used as value for each time-lapse
        condition.

    Returns
    -------
    Dict[str, Dict[str, np.ndarray]]
        Survival function data for specified time-lapse conditions with the following
        data structure:
            {
                "t_tl": {
                    "time": np.ndarray with the time points,
                    "value": np.ndarray with the survival function values,
                },
                ...
            }
    """
    # Dictionary with all the simulated data points
    data = dict()

    if isinstance(N, Sequence):
        if len(N) != len(t_tl_all):
            raise ValueError(
                f"Number of `N` values provided ({len(N)}) is not equal to the number "
                f"of provided time-lapse times provided in `t_tl_all` ({len(t_tl_all)})."
            )
    else:
        N = [N for _ in range(len(t_tl_all))]

    # TODO: future - allow this to be done in a multiprocessed way
    for i, t_tl in enumerate(t_tl_all):
        if isinstance(t_tl, str):
            t_tl_s = data_utils.get_time_sec(t_tl)
        elif isinstance(t_tl, float):
            t_tl_s = t_tl

        data_single = tl_simulation_single(k, s, kb, t_int, t_tl_s, N[i])

        data = {**data, **data_single}

    return data
