"""
Module for GRID and multi-exponential fitting.
"""

import math
from typing import Tuple

import numpy as np
import scipy.optimize

import gridlib.calc as calc
import gridlib.data_utils as data_utils

# TODO: I think t_int should not be set but should be inferred, discuss with Ihor?


def create_fixed_grid(
    k_min: float, k_max: float, n: int, scale: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Function creates the fixed grid based on the provided parameters.

    Parameters
    ----------
    k_min: float
        Minimum decay rate value.
    k_max: float
        Maximum decay rate value.
    n: int
        Number of positions in the grid.
    scale: str
        The scale to be applied for the creation of the grid.
        Options:
            "log": logarithmic scale
            "linear": linear scale

    Returns
    -------
    k: np.ndarray
        The fixed grid positions based on the given parameter values.
    """

    if scale not in {"log", "linear"}:
        raise ValueError("Scale should be either 'log' or 'linear'.")

    # Create the grid with fixed spacing between every position
    if scale == "linear":
        k = np.linspace(k_min, k_max, num=n, endpoint=True)
    elif scale == "log":
        k = np.geomspace(k_min, k_max, num=n, endpoint=True)

    return k


# Module for GRID fitting, multi-exponential fitting, and GRID resampling
def con_eq(x0: np.ndarray):
    """Constraint equality: makes sure all the amplitudes sum to 1."""
    s = x0[:-1]
    return np.sum(s) - 1


def fit_grid(parameters, data):

    data_processed = data_utils.process_data(data)

    if not data_utils.isvalid_parameters(parameters):
        raise ValueError("parameters does not contain a valid value")

    t_int = parameters["t_int"]

    k = create_fixed_grid(
        parameters["k_min"], parameters["k_max"], parameters["N"], parameters["scale"]
    )

    reg_weight = parameters["reg_weight"]

    x0 = np.ones(k.shape[0], dtype=np.float64) * (1 / k.shape[0])

    # Update the x0 array and set the bounds for both the amplitudes and the
    # photobleaching number
    lbq = 0  # lower bound photobleaching
    ubq = 3  # upper bound photobleaching

    bnds = [(0, 1) for _ in range(x0.shape[0])]
    bnds.append((lbq, ubq))  # add the bounds for the photobleaching
    if parameters["fit_a"]:
        # The initial kb guess is 2 s^-1 as default
        a = 1.0 * t_int  # # default bleaching rate is 1 s^-1
        x0 = np.concatenate((x0, np.array([a], dtype=np.float64)))

    elif not parameters["fit_a"]:
        a = parameters["a_fixed"]
        x0 = np.concatenate((x0, np.array([a])))
        bnds[-1] = (a, a)  # Fix the photobleaching number

    # print(x0.shape)
    # print(bnds)

    cons = [{"type": "eq", "fun": con_eq}]
    # TODO: change iprint, see source code
    options = {"maxiter": 1000, "disp": True, "iprint": 1, "ftol": 10 ** (-10)}

    res = scipy.optimize.minimize(
        calc.lsqobj_grid,
        x0,
        args=(data_processed, k, reg_weight, t_int),
        method="SLSQP",
        jac=True,  # jac=True, means the gradient is given by the function
        bounds=bnds,
        constraints=cons,
        options=options,
    )

    grid_results = {
        "k": k,
        "s": res.x[: k.shape[0]],
        "a": res.x[-1],
        "loss": res.fun,
    }

    fit_results = {"grid": grid_results}

    return fit_results


# TODO: function works!! Yayy, but need to clean it up.


def _fit_n_exp(parameters, data, n: int = 2):
    """Function to fit an n-exponential to the provided data and returns the fitted
    parameter results."""

    data_processed = data_utils.process_data(data)

    if not data_utils.isvalid_parameters(parameters):
        raise ValueError("parameters does not contain a valid value")

    t_int = parameters["t_int"]

    k_min = parameters["k_min"]
    k_max = parameters["k_max"]

    coordinates = []
    for i in range(n):
        coordinates.append(
            np.geomspace(k_min, k_max, num=2, endpoint=True, dtype=np.float64)
        )

    # Create the grid
    x0_k_all = np.meshgrid(*coordinates)
    x0_k_all = np.reshape(x0_k_all, (-1, n))

    # Initial guesses
    x0_s = np.linspace(0.05, 0.7, num=n, endpoint=True, dtype=np.float64)

    # Create the arrays for the bounds, +1 for the photobleaching bounds
    lb = np.zeros(x0_k_all.shape[1] + x0_s.shape[0] + 1)
    ub = np.ones(x0_k_all.shape[1] + x0_s.shape[0] + 1)

    # Set the bounds for the decay rates
    lb[: x0_k_all.shape[1]] = k_min
    ub[: x0_k_all.shape[1]] = k_max

    # Bounds for the amplitudes are already set, since they are the
    # default values, see initialization of the lb and ub array
    # # Set the bounds for the amplitudes
    # lb[x0_k_all.shape[1] : (x0_k_all.shape[1] + x0_s.shape[0])] = 0
    # ub[x0_k_all.shape[1] : (x0_k_all.shape[1] + x0_s.shape[0])] = 1

    # Default values for photobleaching bounds
    lbq = 0  # lower bound photobleaching
    ubq = 3  # upper bound photobleaching
    if parameters["fit_a"]:
        a = 1.0 * t_int  # default bleaching rate is 1 s^-1
        lb[-1] = lbq
        ub[-1] = ubq
    elif not parameters["fit_a"]:
        a = parameters["a_fixed"]
        lb[-1] = parameters["a_fixed"]
        ub[-1] = parameters["a_fixed"]

    bnds = (lb, ub)

    print(f"Total fits: {x0_k_all.shape[0]}")

    fit_results_all = []

    for i in range(x0_k_all.shape[0]):
        x0 = np.concatenate((x0_k_all[i, :], x0_s, np.array([a])))
        res = scipy.optimize.least_squares(
            calc.global_multi_exp,
            x0,
            args=(data_processed, n),
            bounds=bnds,
            method="trf",
        )
        fit_results_all.append(res)

    # Find the best fitted parameters
    cost_min = math.inf
    x_best = None

    for res in fit_results_all:
        if res.cost < cost_min:
            cost_min = res.cost
            x_best = res.x

    # Unpack the k array
    k_temp = x_best[:n]

    # Renormalize s
    s_temp = x_best[n : (2 * n)]
    s_norm = s_temp / np.sum(s_temp)

    # The indices to sort the k_array from low to high
    idx = np.argsort(k_temp)

    multi_exp_results = {
        "k": k_temp[idx],
        "s": s_norm[idx],
        "a": x_best[-1],
        "loss": cost_min,
    }

    fit_results = {f"{n}-exp": multi_exp_results}

    return fit_results


def fit_multi_exp(parameters, data):

    if not data_utils.isvalid_parameters(parameters):
        raise ValueError("parameters does not contain a valid value")

    n_values = parameters["n_exp"]

    fit_results = dict()

    for n in n_values:
        fit_result_n_exp = _fit_n_exp(parameters, data, n=n)
        fit_results = {**fit_results, **fit_result_n_exp}

    return fit_results


def fit(parameters, data):
    """Function to fit both with GRID and with multi-exponentials."""

    fit_results_grid = fit_grid(parameters, data)
    fit_results_multi_exp = fit_multi_exp(parameters, data)

    fit_results = {**fit_results_grid, **fit_results_multi_exp}

    return fit_results
