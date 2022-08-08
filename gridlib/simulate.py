"""
Module with function to sample from a provided GRID spectrum or multi-exponential.
"""
from typing import Sequence, Union
import numpy as np

from . import data_utils


def tl_simulation_single(
    k: Sequence[float],
    s: Sequence[float],
    kb: float,
    t_int: float,
    t_tl: float,
    N: int = 10000,
):

    p = s / np.sum(s)

    # Can also change kb * t_int to a
    keff = (kb * t_int) / t_tl + k

    count = 0
    time = np.linspace(t_tl, 1000 * t_tl, num=1001, endpoint=True)

    binding = np.zeros(time.shape[0])
    binding_sum = 0

    # TODO: this can likely be improved a lot with respect to speed
    rng = np.random.default_rng()
    while binding_sum < N and count < (N * 10):

        keff_state = rng.choice(keff, p=p)
        lifetime = rng.exponential(scale=1 / keff_state)

        if lifetime > time[0]:
            idx = np.sum(lifetime > time) - 1  # - 1 because indexing starts at 0
            binding[idx] = binding[idx] + 1
            binding_sum = binding_sum + 1
        else:
            count = count + 1

    # Calculate the survival function values
    value = np.cumsum(binding[::-1])[::-1]

    # Delete the empty elements
    time = time[value > 0]
    value = value[value > 0]

    return {f"{t_tl}s": {"time": time, "value": value}}


def tl_simulation(
    k: Sequence[float],
    s: Sequence[float],
    kb: float,
    t_int: float,
    t_tl_all: Sequence[Union[float, str]],
    N: Union[int, Sequence] = 100000,
):
    # TODO function to perform multiple simulations and get back a data dictionary
    data = dict()

    if isinstance(N, Sequence):
        if len(N) != len(t_tl_all):
            raise ValueError(
                f"Number of `N` values provided ({len(N)}) is not equal to the number "
                f"of provided time-lapse times provided in `t_tl_all` ({len(t_tl_all)})."
            )
    else:
        N = [N for _ in range(len(t_tl_all))]

    for i, t_tl in enumerate(t_tl_all):
        if isinstance(t_tl, str):
            t_tl_s = data_utils.get_time_sec(t_tl)
        elif isinstance(t_tl, float):
            t_tl_s = t_tl

        data_single = tl_simulation_single(k, s, kb, t_int, t_tl_s, N[i])

        data = {**data, **data_single}

    data = data_utils.fmt_t_str_data(data)

    return data
