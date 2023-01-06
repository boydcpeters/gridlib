"""
Module with function to sample from a provided GRID spectrum or multi-exponential.
"""
import math
from typing import Sequence, Union, Dict

import numpy as np

from . import data_utils


def tl_simulation_single(
    k: np.ndarray,
    s: np.ndarray,
    kb: float,
    t_int: float,
    t_tl: float,
    N: int = 10000,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Function simulates the fluorescence survival time distributions of a molecule for
    one time-lapse conditions. Dissociation and photobleaching of molecules with
    user-defined parameters is simulated and the resulting fluorescence survival time
    distribution is calculated.

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
        A dictionary mapping keys (time-lapse conditions) to the corresponding time and
        value arrays of the survival functions. For example::

            {
                "0.05s": {
                    "time": array([0.05, 0.1, 0.15, ...])
                    "value": array([1.000e+04, 8.464e+03, 7.396e+03, ...])
                },
            }

    Raises
    ------
    ValueError
        If kb, t_int, t_tl or N is incorrectly set. They should all atleast be larger
        than 0.

    Examples
    --------
    >>> data_sim = tl_simulation_single(np.array([0.005, 0.48, 5.2]),
    ... np.array([0.05, 0.25, 0.7]), 0.03, 0.05, 0.5, N=10000)

    The above function call simulates fluorescence survival time distributions of a
    type of molecule with three different dissociation rates, and a photobleaching
    rate of 0.03 s^-1. The integration time is set to 50 ms and the time-lapse time
    is set to 500 ms. For the survival time distribution 10000 observed molecules are
    simulated.
    """

    if not math.isclose(1.0, np.sum(s)):
        raise ValueError(f"Sum of amplitudes should be equal to 1, but is {np.sum(s)}.")

    if kb <= 0:
        raise ValueError(
            "Photobleaching rate ('kb' argument) should be larger than zero"
        )
    if t_int <= 0:
        raise ValueError(
            "Integration time ('t_int' argument) should be larger than zero"
        )
    if t_tl <= 0:
        raise ValueError("Time-lapse time ('t_tl' argument) should be larger than zero")
    if N <= 0:
        raise ValueError(
            "Number of simulated molecules ('N' argument) should be larger than zero."
        )

    BUFFER_SIZE_DEFAULT = 1000

    p = s / np.sum(s)

    # Can also change kb * t_int to a
    keff = (kb * t_int) / t_tl + k

    time = np.linspace(0, 1500 * t_tl, num=1501, endpoint=True)

    binding = np.zeros(time.shape[0])
    binding_sum = 0

    # TODO: future - improve speed: multiprocessing, or use "chunks", so do like
    # 1000 molecules at the same time with arrays?
    rng = np.random.default_rng()

    # Simulate until enough points have been captured or the computation time is
    # exceeded
    buffer_size = min(BUFFER_SIZE_DEFAULT, N)

    count = 0
    max_count = 1000 * N
    while binding_sum < N and count < max_count:

        keff_state = rng.choice(keff, size=(buffer_size,), p=p)
        lifetime = rng.exponential(scale=1 / keff_state)

        idx = lifetime // t_tl
        idx = idx.astype(np.int32)

        idx_uniq, idx_count = np.unique(idx, return_counts=True)

        binding[idx_uniq] = binding[idx_uniq] + idx_count

        # Only count the molecules that are bound for at least one delta t
        binding_sum = np.sum(binding) - binding[0]

        # Keep track of the number of simulations ran
        count = count + buffer_size

        # Reset the buffer size if required
        buffer_size = min(BUFFER_SIZE_DEFAULT, int(round(N - binding_sum, 0)))

    if count >= max_count:
        print(
            "The simulation was finished pre-emptively because max_count",
            "(1000*N) value was reached.",
        )

    # Remove the the molecules that are bound for 0 frames
    binding = binding[1:]
    # Remove the first time point as well
    time = time[1:]

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
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Function simulates the fluorescence survival time distributions of a molecule for
    different time-lapse conditions. Dissociation and photobleaching of molecules with
    user-defined parameters is simulated and the resulting fluorescence survival time
    distribution is calculated. This is done for all the different user-specified
    time-lapse conditions.

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
        A dictionary mapping keys (time-lapse conditions) to the corresponding time and
        value arrays of the survival functions. For example::

            {
                "0.05s": {
                    "time": array([0.05, 0.1, 0.15, ...]),
                    "value": array([1.000e+04, 8.464e+03, 7.396e+03, ...]),
                },
                "1s": {
                    "time": array([1., 2., 3., ...]),
                    "value": array([1.000e+04, 6.925e+03, 5.541e+03, ...]),
                },
            }

    Raises
    ------
    ValueError
        If N is a list and does not have the same length as t_tl_all list.

    Examples
    --------
    >>> data_sim = tl_simulation(np.array([0.005, 0.48, 5.2]),
    ... np.array([0.05, 0.25, 0.7]), 0.03, 0.05, [0.05, 0.2, 1.0, 5.0], N=10000)

    The above function call simulates fluorescence survival time distributions of a
    type of molecule with three different dissociation rates, and a photobleaching
    rate of 0.03 s^-1. The integration time is set to 50 ms and the time-lapse times
    are set to 50 ms, 200 ms, 1 s and 5 s. For each survival time distribution,
    10000 observed molecules are simulated.

    >>> data_sim = tl_simulation(np.array([0.005, 0.48, 5.2]),
    ... np.array([0.05, 0.25, 0.7]), 0.03, 0.05, [0.05, 0.2, 1.0, 5.0],
    ... N=[10000, 5000, 2500, 1500])

    Same as the first example, but now there is a different number of molecules
    simulated for every time-lapse condition.
    )
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
