"""
Module with function to sample from a provided GRID spectrum or multi-exponential.
"""
import matplotlib.pyplot as plt
import numpy as np


def choose_state(keff: np.ndarray, p: np.ndarray):
    """Function returns the keff associated to a certain state depending on p"""
    return np.random.choice(keff, p=p)


def tl_simulation_single(t_int, t_tl, kb, k, N: int = 10000):

    # TODO: this is temporary, should be provided or inferred?
    p = np.ones(k.shape)  # what is the probability of k, ie amplitude
    p = p / np.sum(p)

    # Can also change kb * t_int to a
    keff = ((kb * t_int) / t_tl) + k

    count = 0
    time = np.linspace(t_tl, 1000 * t_tl, num=1001, endpoint=True)
    binding = np.zeros(time.shape[0])

    rng = np.random.default_rng()
    while np.sum(binding) < N and count < 10**7:

        keff_state = choose_state(keff, p)
        lifetime = rng.exponential(scale=1 / keff_state)

        if lifetime > time[0]:
            binding[np.sum(lifetime > time)] = binding[np.sum(lifetime > time)] + 1
        else:
            count = count + 1

    # Calculate the survival function values
    value = np.cumsum(binding[::-1])[::-1]

    # Delete the empty fields
    time = time[value > 0]
    value = value[value > 0]

    return {f"{t_tl}": {"time": time, "value": value}}


def tl_simulations(
    t_tls,
):
    # TODO function to perform multiple simulations and get back a data dictionary
    pass
