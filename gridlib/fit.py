from typing import Tuple
import numpy as np
import scipy.optimize

import gridlib.data_utils as data_utils
import gridlib.calc as calc

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
        x0 = np.concatenate((x0, np.array([0.01], dtype=np.float64)))

    elif not parameters["fit_a"]:
        x0 = np.concatenate((x0, np.array([parameters["a_fixed"]])))
        bnds[-1] = (parameters["a_fixed"], parameters["a_fixed"])

    # print(x0.shape)
    # print(bnds)

    cons = [{"type": "eq", "fun": con_eq}]
    options = {"maxiter": 1000, "disp": True}

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

    grid_results = {"k": k, "s": res.x[: k.shape[0]], "a": res.x[-1]}

    fit_results = {"grid": grid_results}

    return fit_results


def fit_multi_exp(parameters, data):
    """Function to fit a multi-exponential to the provided data and returns the fitted
    parameter results."""
    pass


def fit(parameters, data):
    """Function to fit both with GRID and with a multi-exponential."""
    pass