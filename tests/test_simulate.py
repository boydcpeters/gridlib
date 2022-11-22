"""Tests for `calc` module."""
import numpy as np
import pytest

from gridlib import simulate

# TODO: write tests
# def test_tl_simulations():
#     """Test the working of the `tl_simulate_single` function."""
#     k = np.array([0.005, 0.1, 4.2], dtype=np.float64)
#     s = np.array([0.04, 0.2, 0.76], dtype=np.float64)
#     kb = 1.0
#     t_int = 0.05
#     t_tl_all = [0.05, 0.2, 1.0, 5.0]  # , 9.0]
#     N = 10000

#     data = simulate.tl_simulations(k, s, kb, t_int, t_tl_all, N)
#     print(data.keys())


def test_tl_simulation_single():

    k_correct = np.array([0.01, 0.5, 1.4, 5.9])
    s_correct = np.array([0.02, 0.1, 0.25, 0.63])
    kb_correct = 0.2
    t_int_correct = 0.05
    t_tl_correct = 1.0
    N_correct = 1000

    assert (
        simulate.tl_simulation_single(
            k_correct, s_correct, kb_correct, t_int_correct, t_tl_correct, N=N_correct
        )
        is not None
    )

    # Test whether an error is thrown if the sum of the weights does not equal 1
    s_zeros = np.zeros(4)
    with pytest.raises(ValueError):
        simulate.tl_simulation_single(
            k_correct, s_zeros, kb_correct, t_int_correct, t_tl_correct, N=N_correct
        )

    with pytest.raises(ValueError):
        simulate.tl_simulation_single(
            k_correct, s_correct, kb_correct, t_int_correct, t_tl_correct, N=0
        )
    with pytest.raises(ValueError):
        simulate.tl_simulation_single(
            k_correct, s_correct, kb_correct, t_int_correct, t_tl_correct, N=-1
        )
