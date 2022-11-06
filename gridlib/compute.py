"""
Module with functions to compute curves or survival functions.
"""
import math
from typing import Dict, List, Tuple, Union

import numpy as np

from . import calc


def retrieve_track_lifes(data, filtered: bool = True) -> Dict:
    """Function retrieves the amount of frames a molecule is bound.

    Parameters
    ----------
    data : List[List]
        The data values as follows: (f, t, x, y, track_id, disp, intensity, sigma, fit_error)
    filtered : bool (CURRENTLY NOT IMPLEMENTED!!!)
        Only return residence frame lengths when the full tracklet is imaged. You don't know
        how long a particle has been bound at the start of imaging or how long it will stay bound
        after the imaging stops, so these particle residence frame length are not really
        trustworthy.

    Returns
    -------
    track_lifes: np.array
    """
    # TODO: filtering is currently not implemented
    track_lifes = []

    frame_start = None
    frame_end = None

    for i in range(len(data)):

        # if there is no value for the frame_start and frame_end, set a value, ie. first data point
        if frame_start is None:
            frame_start = data[i][0]
            frame_end = data[i][0]
        elif data[i][4] == data[i - 1][4]:
            frame_end = data[i][0]
        else:
            # If frame_end=5, and frame_start=2, then track_life should be 3
            # ie. count the amount of delta t's between the start and end
            track_life = frame_end - frame_start
            track_lifes.append(track_life)
            frame_start = data[i][0]
            frame_end = data[i][0]

        # if it is the last data point, add the track life
        if i == len(data) - 1:
            track_life = frame_end - frame_start
            track_lifes.append(track_life)

    return track_lifes


# TODO: update docstring
def compute_survival_function(
    track_lifes: Union[List, np.ndarray],
    t_tl: float,
    min_track_life: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Function creates the time and value arrays describing the survival function distribution from
    a set of track lifes.


    # If frame_end=5, and frame_start=2, then track_life should be 3
    # ie. count the amount of delta t's between the start and end

    So track lifes is the number of delta t's.

    Parameters
    ----------
    track_lifes : List or np.ndarray
        Contains all the individual track lifes, so duplicate track life values are
        allowed/expected.
    t_tl : float
        Time-lapse time in seconds.
    min_track_life : int
        The minimum track life length, this is inferred when the value is set to None, and then it
        is assumed that the minimum value in the track_lifes list/np.ndarray is equal to the minimum
        allowable value set in the settings. However, this can also be set manually. (default None)

    Returns
    -------
    time: np.ndarray
        The time vector indicating all the time points for the values of the survival function.
    sf_value: np.ndarray
        The survival function values for their respective time point.
    """
    if isinstance(track_lifes, list):
        track_lifes = np.array(track_lifes, dtype=np.int64)

    if not np.issubdtype(track_lifes.dtype, np.integer):
        raise ValueError("The dtype of the provided array should be integer.")

    if min_track_life is None:
        # Get the shortest possible track life from all the track lifes. min_track_length assumes
        # that the minimum allowable track length is in the dataset, so if the minimum track length
        # was set to 3 than we assume that a track life of 3 is at least in the dataset once,
        # otherwise the min_track_length should be set in the function call.
        min_track_life = np.amin(track_lifes)

    # Get the largest possible track in this tl condition
    max_track_life = np.amax(track_lifes)

    # Retrieve the unique track length values and their counts
    track_life_values, count = np.unique(track_lifes, return_counts=True)

    # Array to store the number of occurences of each lifetime.
    amount = np.zeros(max_track_life + 1, dtype=np.int64)

    # Fill the array at the indices of the the unique track lengths with their respective count
    amount[track_life_values] = count

    # Delete the first n rows to account for the min_track_length
    amount = amount[min_track_life:]

    # Create the time vector
    time = np.linspace(
        min_track_life * t_tl,
        max_track_life * t_tl,
        num=((max_track_life - min_track_life) + 1),
        dtype=np.float64,
    )

    # Calculate the survival function values
    value = np.cumsum(amount[::-1])[::-1]

    return time, value


def compute_multi_exp(
    k: np.ndarray, s: np.ndarray, a: float, time: np.ndarray
) -> np.ndarray:
    """Function computes the multi-exponential for the given time points.

    Parameters
    ----------
    k : np.ndarray
        Decay rates
    s : np.ndarray
        Amplitudes of the respective decay rates. The sum of the amplitudes should
        be equal to zero.
    a : float
        Photobleaching number (a * t_int)
    time : np.ndarray
        Time points for which the multi-exponential needs to be determined.

    Returns
    -------
    value : np.ndarray
        The resulting normalized multi-exponential values at the given time points.
    """

    if not math.isclose(1.0, np.sum(s)):
        raise ValueError(f"Sum of amplitudes should be equal to 1, but is {np.sum(s)}.")

    # Frame number, start counting at 1
    m = np.arange(time.shape[0]) + 1

    # Calculate the model function
    # Decay spectrum
    h = calc.calch(time, k, s)
    # Bleaching
    q = np.exp(-a * m)

    # Calculate the complete GRID fit function
    value = q / q[0] * h / h[0]

    return value


def compute_grid_curve(
    k: np.ndarray, s: np.ndarray, a: float, time: np.ndarray
) -> np.ndarray:
    """Function computes the GRID curve for the given time points. This is just a
    wrapper function for :func:`compute_multi_exp`, since the GRID curve is just a
    multi-exponential curve.

    Parameters
    ----------
    k : np.ndarray
        Decay rates
    s : np.ndarray
        Amplitudes of the respective decay rates. The sum of the amplitudes should
        be equal to zero.
    a : float
        Photobleaching number (a * t_int)
    time : np.ndarray
        Time points for which the multi-exponential needs to be determined.

    Returns
    -------
    value : np.ndarray
        The resulting normalized multi-exponential values at the given time points.
    """

    value = compute_multi_exp(k, s, a, time)

    return value


def compute_multi_exp_for_data(
    fit_values_multi_exp: Dict[str, Union[np.array, float]],
    data: Dict[str, Dict[str, np.ndarray]],
):
    """Function computes the multi-exponential curves given the required parameters.

    Parameters
    ----------
    fit_values_multi_exp: Dict[str, Union[np.array, float]]
        Dictionary with the following key-value pairs:
            "k": np.ndarray with the dissociation rates
            "s": np.ndarray with the corresponding weights
            "a": bleaching number (a = k_b * t_int)
            Optional:
                if GRID fit:
                "fit_error": error of the fit
                if multi-exponential fit:
                "Adj_R_squared": adjusted R squared for every k and S
    data: Dict[str, Dict[str, np.ndarray]]
        Survival function data for time-lapse conditions with the following data
        structure:
            {
                "t_tl": {
                    "time": np.ndarray with the time points,
                    "value": np.ndarray with the survival function values,
                },
                ...
            }

    Returns
    -------
    data_multi_exp: Dict[str, Dict[str, np.ndarray]]
        Survival function data computed with the provided multi-exponential fit values
        for every time-lapse conditions in data. The object has the following data
        structure:
            {
                "t_tl": {
                    "time": np.ndarray with the time points,
                    "value": np.ndarray with the survival function values,
                }
            }
    """
    k = fit_values_multi_exp["k"]
    s = fit_values_multi_exp["s"]
    a = fit_values_multi_exp["a"]

    # Dictionary to store the GRID fit results
    data_multi_exp = dict()

    for t_tl in data.keys():

        data_multi_exp[t_tl] = dict()

        # Time points
        t = data[t_tl]["time"]

        # Store the time points
        data_multi_exp[t_tl]["time"] = t

        data_multi_exp[t_tl]["value"] = compute_multi_exp(k, s, a, t)

    return data_multi_exp


def compute_grid_curves_for_data(
    fit_values_grid: Dict[str, Union[np.array, float]],
    data: Dict[str, Dict[str, np.ndarray]],
):
    """Function computes the GRID curves for the given fit_values_grid parameters.
    This is a convenience function and is just a wrapper around the function
    :func:`calc_multi_exp`.

    Parameters
    ----------
    fit_values_grid: Dict[str, Union[np.array, float]]
        Dictionary with the following key-value pairs:
            "k": np.ndarray with the dissociation rates
            "s": np.ndarray with the corresponding weights
            "a": bleaching number (a = k_b * t_int)
            Optional:
                "fit_error": error of the fit
    data: Dict[str, Dict[str, np.ndarray]]
        Survival function data for time-lapse conditions with the following data
        structure:
            {
                "t_tl": {
                    "time": np.ndarray with the time points,
                    "value": np.ndarray with the survival function values,
                },
                ...
            }

    Returns
    -------
    data_grid: Dict[str, Dict[str, np.ndarray]]
        Survival function data computed with the provided GRID fit values for every
        time-lapse conditions in data. The object has the following data structure:
            {
                "t_tl": {
                    "time": np.ndarray with the time points,
                    "value": np.ndarray with the survival function values,
                },
                ...
            }
    """

    data_grid = compute_multi_exp_for_data(fit_values_grid, data)

    return data_grid
