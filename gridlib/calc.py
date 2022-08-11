"""
Module with all the functions required for the GRID and multi-exponential
calculations.
"""
from typing import Dict, Tuple, Union

import numpy as np
from numba import njit

# TODO: Make lsq_obj_multi_exp() (with n being the number of exponents in the wrapper function)


@njit()
def transmat(time: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Function to calculate the forward Laplace transformation matrix A (MxN)-
    matrix.

    Parameters
    ----------
    time: np.ndarray
        time points (m * t_tl)
    k: np.ndarray
        decay rates

    Returns
    -------
    A: np.ndarray
        Forward Laplace transformation matrix A (MxN) matrix
    """

    time = np.reshape(time, (time.shape[0], 1))
    k = np.reshape(k, (1, k.shape[0]))

    A = np.exp(-(time @ k))

    # m = time.shape[0]
    # n = k.shape[0]
    # A = np.ones((m, n))
    #
    # for i in range(m):
    #     for j in range(n):
    #         A[i, j] = np.exp(-k[j]*time[i])

    return A


@njit()
def calch(time: np.ndarray, k: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Function calculates the superposition of the different independent decay
    curves with the forward Laplace transformation.

    Parameters
    ----------
    time: np.ndarray
        The time points to calculate the forward Laplace transform for.
    k: np.ndarray
        Decay rates.
    s: np.ndarray
        Amplitudes of the decay rates.

    Returns
    -------
    h: np.ndarray
        Superposition of all the independent decay processes for the time points time.
    """

    # Transformation matrix
    A = transmat(time, k)

    # Weighted transformation
    h = A @ s

    return h


@njit()
def gradh(
    eq0: np.ndarray, q: np.ndarray, h: np.ndarray, time: np.ndarray, k: np.ndarray
) -> np.ndarray:
    """Function calculates the gradient for the amplitudes of the decay rates.

    Parameters
    ----------
    eq0: np.ndarray
        Array with the distances between the model and the data.
    q: np.ndarray
        Array with values resulting from photobleaching.
    h: np.ndarray
        Array with values resulting from independent decay processes.
    time: np.ndarray
        Array with all the time points for which the values need to be calculated.
    k: np.ndarray
        Array with all the decay rates.
    Returns
    -------
    grad_s: np.ndarray
        Values for the gradient of the amplitudes for the respective decay rates in k.
    """

    # Derivatie with respect to Si
    A = transmat(time, k)
    A = A.T

    grad_s = np.zeros(k.shape[0])

    # Derivative
    for m in range(time.shape[0]):
        grad_s = grad_s + 2 * eq0[m] * q[m] / q[0] * (
            A[:, m] / h[0] - A[:, 0] * h[m] / (h[0]) ** 2
        )

    return grad_s


@njit()
def gradq(
    eq0: np.ndarray, q: np.ndarray, h: np.ndarray, time: np.ndarray, a: float
) -> float:
    """Function calculates the gradient for the photobleaching.

    Parameters
    ----------
    eq0: np.ndarray
        Array with the distances between the model and the data.
    q: np.ndarray
        Array with values resulting from photobleaching.
    h: np.ndarray
        Array with values resulting from independent decay processes.
    time: np.ndarray
        Array with all the time points for which the values need to be calculated.
    a: np.ndarray
        Photobleaching number (k_b * t_int)

    Returns
    -------
    grad_a: float
        Gradient value for the photobleaching number.
    """

    grad_a = 0

    for m in range(eq0.shape[0]):
        grad_a = grad_a + 2 * eq0[m] * (h[m] / h[0]) * -m * np.exp(-a * m)
    return grad_a


def lsqobj_grid(
    values: np.ndarray,
    data: Dict[str, Dict[str, np.ndarray]],
    k: np.ndarray,
    reg_weight: float,
    t_int: float,
) -> Tuple[float, np.ndarray]:
    """Returns the loss value and the gradient for the GRID fit.

    Objective function for determining dissociation rate spectra. The cost function d
    contains the difference between the GRID fit and measured values and the
    regularization for the mean decay rate of the population.

    Parameters
    ----------
    values : np.ndarray
        Array with decay rate amplitudes + photobleaching number with the structure:
            values = np.array([s, a]) where s is an array with the amplitudes.
    data : Dict[str, Dict[str, np.ndarray]]
        Data of the survival function from real data with the following data structure:
        {
            "t_tl": {
                "time": np.ndarray with the time points,
                "value": np.ndarray with the survival function values,
            }
        }
    k : np.ndarray
        Array of the decay rates of the fixed grid positions
    reg_weight : float
        Regularization weight
    t_int : float
        Integration time

    Returns
    -------
    d : float
        Loss value for the provided values.
    grad: np.ndarray
        Gradient for the provided values.
    """

    # Initialization of the variables
    # ---------------------------------------------------------------------------------
    d = 0
    ceq = 0  # Constrain equality
    gradceq = np.zeros(values.shape[0], dtype=np.float64)
    gradreg = np.zeros(values.shape[0], dtype=np.float64)

    # Unpack the values array and assign them to the correct variable
    # ---------------------------------------------------------------------------------
    s = values[: k.shape[0]]
    a = values[k.shape[0]]

    # Constraints for global fit
    # ---------------------------------------------------------------------------------

    # Cost function for difference between fit and measurement
    # Loop over all the time-lapse conditions
    for t_tl in data.keys():
        # read n-th measured time-lapse
        time = data[t_tl]["time"]  # time-points (lifetime) array
        p = data[t_tl]["value"]  # probability array

        # IMPORTANT: frame count should start at 1, NOT 0
        m = np.arange(time.shape[0]) + 1

        # Calculate the model function
        # Decay spectrum
        h = calch(time, k, s)
        # Photobleaching
        q = np.exp(-a * m)

        # Calculate the squared differences
        eq0 = ((q / q[0]) * (h / h[0])) - (p / p[0])
        ceqt = eq0**2

        # Generate gradient
        gradhl = gradh(eq0, q, h, time, k)
        gradql = gradq(eq0, q, h, time, a)
        gradceqt = np.concatenate((gradhl, np.array([gradql])))

        # Bind constraint and gradient
        ceq = ceq + np.sum(ceqt)
        gradceq = gradceq + gradceqt

    # Regularization for the dead time
    # ------------------------------------------------------------------
    # Initialize variables
    tau = np.array([t_int, 2 * t_int], dtype=np.float64)  # death time
    A = transmat(tau, k)

    # Calculate the mean decay rate at borders t
    kquer = (A @ (k * s)) / (A @ s)

    # Regularisation
    # TODO: check the 0.5, does not seem to be in the GRID paper
    reg = 0.5 * (kquer[0] - kquer[1]) ** 2

    # Gradient of regularisation
    temp0 = A[0, :].T / (A[0, :] @ s) * (k - kquer[0])
    temp1 = A[1, :].T / (A[1, :] @ s) * (k - kquer[1])
    gradreg[0 : s.shape[0]] = (kquer[0] - kquer[1]) * (temp0 - temp1)

    # Calculate the complete cost function
    # -------------------------------------------------------------------
    d = ceq + reg_weight * reg
    grad = gradceq + reg_weight * gradreg

    return d, grad


def global_multi_exp(
    values: np.ndarray,
    data: Dict[str, Dict[str, np.ndarray]],
    n: int,
) -> Tuple[float, np.ndarray]:
    """Function returns the residuals"""
    # Initialization of the variables
    # ---------------------------------------------------------------------------------
    d = np.array([], dtype=np.float64)

    # Unpack the values array and assign them to the correct variable
    # ---------------------------------------------------------------------------------
    k = values[:n]
    s = values[n : (2 * n)]
    a = values[-1]

    # Constraints for global fit
    # ---------------------------------------------------------------------------------

    # Cost function for difference between fit and measurement
    # Loop over all the time-lapse conditions
    for t_tl in data.keys():
        # read n-th measured time-lapse
        time = data[t_tl]["time"]  # time-points (lifetime) array
        p = data[t_tl]["value"]  # probability array

        # IMPORTANT: frame count should start at 1, NOT 0
        m = np.arange(time.shape[0]) + 1

        # Calculate the model function
        # Decay spectrum
        h = calch(time, k, s)
        # Photobleaching
        q = np.exp(-a * m)

        # Calculate the residuals
        eq0 = ((q / q[0]) * (h / h[0])) - (p / p[0])

        d = np.concatenate((d, eq0))

    return d
