"""
Module for GRID and multi-exponential fitting.
"""

import math
from typing import Tuple

import numpy as np
import scipy.optimize

import gridlib.calc as calc
import gridlib.data_utils as data_utils


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

    if not data_utils.isvalid_parameters_grid(parameters):
        raise ValueError("parameters does not contain a valid value")

    if "k" in parameters:
        k = parameters["k"]
    else:
        k = create_fixed_grid(
            parameters["k_min"],
            parameters["k_max"],
            parameters["N"],
            parameters["scale"],
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

        tau_min = min(data_utils.get_time_sec(t_tl) for t_tl in data.keys())
        # The initial kb guess is 2 s^-1 as default
        a = 1.0 * tau_min  # # default bleaching rate is 1 s^-1
        x0 = np.concatenate((x0, np.array([a], dtype=np.float64)))

    elif not parameters["fit_a"]:
        a = parameters["a_fixed"]
        x0 = np.concatenate((x0, np.array([a])))
        bnds[-1] = (a, a)  # Fix the photobleaching number

    # print(x0.shape)
    # print(bnds)

    cons = [{"type": "eq", "fun": con_eq}]
    # TODO: change iprint, see source code
    options = {"maxiter": 600, "disp": True, "iprint": 1, "ftol": 10 ** (-11)}

    res = scipy.optimize.minimize(
        calc.lsqobj_grid,
        x0,
        args=(data_processed, k, reg_weight),
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


def _fit_n_exp(parameters, data, n: int = 2):
    """Function to fit an n-exponential to the provided data and returns the fitted
    parameter results."""

    data_processed = data_utils.process_data(data)

    if not data_utils.isvalid_parameters_n_exp(parameters):
        raise ValueError("parameters does not contain a valid value")

    k_min = parameters["k_min"]
    k_max = parameters["k_max"]

    coordinates = []
    for i in range(n):
        coordinates.append(
            np.geomspace(k_min, k_max, num=2, endpoint=True, dtype=np.float64)
        )

    # Create the grid
    x0_k = np.meshgrid(*coordinates)
    x0_k = np.reshape(x0_k, (-1, n))

    # Initial guesses
    x0_s = np.linspace(0.05, 0.7, num=n, endpoint=True, dtype=np.float64)

    # Create the arrays for the bounds of the decay rates, and amplitude values
    # +1 for the photobleaching bounds
    lb = np.zeros(x0_k.shape[1] + x0_s.shape[0] + 1)
    ub = np.ones(x0_k.shape[1] + x0_s.shape[0] + 1)

    # Set the bounds for the decay rates
    lb[: x0_k.shape[1]] = k_min
    ub[: x0_k.shape[1]] = k_max

    # Bounds for the amplitudes are already set, since they are the
    # default values, see initialization of the lb and ub array
    # # Set the bounds for the amplitudes
    # lb[x0_k.shape[1] : (x0_k.shape[1] + x0_s.shape[0])] = 0
    # ub[x0_k.shape[1] : (x0_k.shape[1] + x0_s.shape[0])] = 1

    # Set the initial photobleaching number and the photobleaching bounds
    if parameters["fit_a"]:
        tau_min = min(data_utils.get_time_sec(t_tl) for t_tl in data.keys())
        # The initial kb guess is 2 s^-1 as default
        a = 1.0 * tau_min  # # default bleaching rate is 1 s^-1

        # Default values for photobleaching bounds
        lbq = 0  # lower bound photobleaching
        ubq = 3  # upper bound photobleaching

        lb[-1] = lbq
        ub[-1] = ubq
    elif not parameters["fit_a"]:
        a = parameters["a_fixed"]
        lb[-1] = parameters["a_fixed"]
        ub[-1] = parameters["a_fixed"]

    bnds = (lb, ub)

    print(f"Total fits: {x0_k.shape[0]}")

    fit_results_all = []

    for i in range(x0_k.shape[0]):
        x0 = np.concatenate((x0_k[i, :], x0_s, np.array([a])))
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

    # Retrieve the best decay rates
    k_temp = x_best[:n]

    # Retrieve the best weight values
    s_temp = x_best[n : (2 * n)]
    # Renormalize s
    s_norm = s_temp / np.sum(s_temp)

    # The indices to sort the k_array from low to high
    idx = np.argsort(k_temp)

    # Store the best results in a dictionary
    multi_exp_results = {
        "k": k_temp[idx],
        "s": s_norm[idx],
        "a": x_best[-1],
        "loss": cost_min,
    }

    # Store the results in a dictionary as a value with the key indicating the
    # number of decay rates
    fit_results = {f"{n}-exp": multi_exp_results}

    return fit_results


def fit_multi_exp(parameters, data):

    if not data_utils.isvalid_parameters_n_exp(parameters):
        raise ValueError("parameters does not contain a valid value")

    n = parameters["n_exp"]

    fit_results = dict()

    if isinstance(n, int):
        fit_result_n_exp = _fit_n_exp(parameters, data, n=n)
        fit_results = {**fit_results, **fit_result_n_exp}
    else:
        for n_value in n:
            fit_result_n_exp = _fit_n_exp(parameters, data, n=n_value)
            fit_results = {**fit_results, **fit_result_n_exp}

    return fit_results


def fit(parameters, data):
    """Function to fit both with GRID and with multi-exponentials."""

    parameters_grid, parameters_n_exp = data_utils.split_parameters(parameters)

    fit_results_grid = fit_grid(parameters_grid, data)
    fit_results_multi_exp = fit_multi_exp(parameters_n_exp, data)

    fit_results = {**fit_results_grid, **fit_results_multi_exp}

    return fit_results


def _fit_n_exp_fixed_k(parameters, data, n: int = 2):
    """Function to fit an n-exponential to the provided data and returns the fitted
    parameter results."""

    data_processed = data_utils.process_data(data)

    if not data_utils.isvalid_parameters(parameters):
        raise ValueError("parameters does not contain a valid value")

    if "k" in parameters:
        x0_k = np.copy(parameters["k"])
        x0_k = np.reshape(x0_k, (-1,))

        n = x0_k.shape[0]
    else:
        k_min = parameters["k_min"]
        k_max = parameters["k_max"]

        coordinates = []
        for i in range(n):
            coordinates.append(
                np.geomspace(k_min, k_max, num=2, endpoint=True, dtype=np.float64)
            )

        # Create the grid
        x0_k = np.meshgrid(*coordinates)
        x0_k = np.reshape(x0_k, (-1, n))

    # Create the arrays for the bounds of the decay rates, and amplitude values
    # +1 for the photobleaching bounds
    lb = np.zeros(x0_k.shape[1] + x0_s.shape[0] + 1)
    ub = np.ones(x0_k.shape[1] + x0_s.shape[0] + 1)

    # Initial guesses
    x0_s = np.linspace(0.05, 0.7, num=n, endpoint=True, dtype=np.float64)

    if "k" in parameters:
        # Set the bounds for the decay rates
        lb[: x0_k.shape[1]] = x0_k
        ub[: x0_k.shape[1]] = x0_k
    else:
        # Set the bounds for the decay rates
        lb[: x0_k.shape[1]] = k_min
        ub[: x0_k.shape[1]] = k_max

    # Bounds for the amplitudes are already set, since they are the
    # default values, see initialization of the lb and ub array
    # # Set the bounds for the amplitudes
    # lb[x0_k.shape[1] : (x0_k.shape[1] + x0_s.shape[0])] = 0
    # ub[x0_k.shape[1] : (x0_k.shape[1] + x0_s.shape[0])] = 1

    # Set the initial photobleaching number and the photobleaching bounds
    if parameters["fit_a"]:
        tau_min = min(data_utils.get_time_sec(t_tl) for t_tl in data.keys())
        # The initial kb guess is 2 s^-1 as default
        a = 1.0 * tau_min  # # default bleaching rate is 1 s^-1

        # Default values for photobleaching bounds
        lbq = 0  # lower bound photobleaching
        ubq = 3  # upper bound photobleaching

        lb[-1] = lbq
        ub[-1] = ubq
    elif not parameters["fit_a"]:
        a = parameters["a_fixed"]
        lb[-1] = parameters["a_fixed"]
        ub[-1] = parameters["a_fixed"]

    bnds = (lb, ub)

    print(f"Total fits: {x0_k.shape[0]}")

    fit_results_all = []

    for i in range(x0_k.shape[0]):
        x0 = np.concatenate((x0_k[i, :], x0_s, np.array([a])))
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

    # Retrieve the best decay rates
    k_temp = x_best[:n]

    # Retrieve the best weight values
    s_temp = x_best[n : (2 * n)]
    # Renormalize s
    s_norm = s_temp / np.sum(s_temp)

    # The indices to sort the k_array from low to high
    idx = np.argsort(k_temp)

    # Store the best results in a dictionary
    multi_exp_results = {
        "k": k_temp[idx],
        "s": s_norm[idx],
        "a": x_best[-1],
        "loss": cost_min,
    }

    # Store the results in a dictionary as a value with the key indicating the
    # number of decay rates
    fit_results = {f"{n}-exp": multi_exp_results}

    return fit_results
