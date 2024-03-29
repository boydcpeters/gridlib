"""
Module for utility functions.
"""

import copy
import math
import re
from typing import Dict

import numpy as np


def _num_decimal_places(value: str) -> int:
    """Function finds the number of decimal places in a string float.

    Parameters
    ----------
    value : str
        A float in string format for which the number of decimals should be determined.

    Returns
    -------
    int
        Number of decimals

    Examples
    --------
    >>> _num_decimal_places("5.32")
    2
    >>> _num_decimal_places("2.")
    0
    >>> _num_decimal_places("98.8572")
    4
    >>> _num_decimal_places("3.4120")
    3
    >>> _num_decimal_places("11")
    0
    >>> _num_decimal_places("0.05")
    2
    """
    # Regex from:
    # https://stackoverflow.com/questions/28749177/how-to-get-number-of-decimal-places
    # m = re.match(r"^[0-9]*\.([1-9]([0-9]*[1-9])?)0*$", value)
    # Regex is adapted such that "0.05" also gets 2 decimal places, regex above
    # requires the first value after the dot being [1-9]
    m = re.match(r"^[0-9]*\.([0-9]*[1-9])0*$", value)
    return len(m.group(1)) if m is not None else 0


def get_time_sec(t_str: str, return_num_decimals: bool = False) -> float:
    """Function retrieves the time in seconds from a time string.

    Parameters
    ----------
    t_str : str
        A value string from which the time in units of seconds needs to be retrieved.
    return_num_decimals : bool, optional
        Specifies whether the number of decimals needs to be returned. (default False)

    Returns
    -------
    t_s : float
        Time in seconds
    d_places : int, optional
        Number of decimal places. Only provided if `return_num_decimals` is True.

    Examples
    --------
    >>> get_time_sec("50ms")
    0.05
    >>> get_time_sec("4.2s")
    4.2
    >>> get_time_sec("1150ms")
    1.15
    """

    if not isinstance(t_str, str):
        raise ValueError("Input argument should be a string.")

    # pattern to match time pattern in the files, eg. 100ms or 2s
    time_pattern = re.compile(
        r"[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)", flags=re.IGNORECASE
    )
    unit_pattern = re.compile("(ms|s)", flags=re.IGNORECASE)

    # Match time and unit pattern
    m_time = time_pattern.search(t_str)
    m_unit = unit_pattern.search(t_str)

    if m_time is None or m_unit is None:
        raise ValueError("t_str is not the correct pattern")

    # get the number value of the match, eg. 50, 100, 390, 4
    t_val_str = m_time.group(0)
    t_val = float(t_val_str)

    # get the unit value of the match, eg. ms or s
    t_unit = m_unit.group(0)

    # Calculate the frame cycle time
    if t_unit == "ms":
        t_s = t_val / 1000.0
        t_s = round(t_s, len(t_val_str))

        # Store it now as a value in the seconds and now find the number of decimal
        # positions when it is stored as seconds
        t_str = f"{t_s:f}s"
        t_s, d_places = get_time_sec(t_str, return_num_decimals=True)

    elif t_unit == "s":
        t_s = t_val
        d_places = _num_decimal_places(t_val_str)
        t_s = round(t_s, d_places)

    if return_num_decimals:
        return t_s, d_places
    else:
        return t_s


def _fmt_t_str_key(t_str: str) -> str:
    t_s, d_places = get_time_sec(t_str, return_num_decimals=True)
    return f"{t_s:.{d_places}f}s"


def fmt_t_str_data(
    data: Dict[str, Dict[str, np.ndarray]]
) -> Dict[str, Dict[str, np.ndarray]]:
    """Function formats all the time-lapse times in the data dictionary to strings
    where the time-lapse times are in seconds. This allows for easier comparison and
    general processing.

    Parameters
    ----------
    data : Dict[str, Dict[str, np.ndarray]]
        Survival function data for time-lapse conditions with the following data
        structure:
            {
                f"{t_tl}": {
                    "time": np.ndarray with the time points,
                    "value": np.ndarray with the survival function values,
                },
                ...
            }

    Returns
    -------
    Dict[str, Dict[str, np.ndarray]]
        Data dictionary but with the time-lapse time keys all in seconds instead of
        seconds or milliseconds.

    Examples
    --------
    >>> data = {"50ms": {"time": np.array([0.05, 0.1, 0.15]), "value": np.array([41, 23, 8])},
                "1s": {"time": np.array([1.0, 2.0]), "value": np.array([34, 9])}}
    >>> fmt_t_str_data(data)
    {"0.05s": {"time": np.array([0.05, 0.1, 0.15]), "value": np.array([41, 23, 8])},
     "1s": {"time": np.array([1.0, 2.0]), "value": np.array([34, 9])}}
    """
    # Deepcopy is required, because otherwise it will store just the reference pointer
    # but not the array itself
    return {_fmt_t_str_key(key): copy.deepcopy(value) for key, value in data.items()}


def process_data(
    data: Dict[str, Dict[str, np.ndarray]], delete: bool = True
) -> Dict[str, Dict[str, np.ndarray]]:
    """Function processes data so functions can be fitted to it. The processing consists
    of three steps. First, the keys of the data are changed to strings with
    the unit being seconds. The second step is optional, but the first data point of
    the shortest time-lapse time is deleted if the first time point is equal to the
    time-lapse time. Finally, the values are normalized so they represent
    probabilities.

    Parameters
    ----------
    data : Dict[str, Dict[str, np.ndarray]]
        Survival function data for time-lapse conditions with the following data
        structure:
            {
                f"{t_tl}": {
                    "time": np.ndarray with the time points,
                    "value": np.ndarray with the survival function values,
                },
                ...
            }
    delete : bool, optional
        Flag to indicate whether the first data point of the shortest time-lapse time
        should be deleted if the first time point is equal to the time-lapse time
        (default True).

    Returns
    -------
    data_processed : Dict[str, Dict[str, np.ndarray]]
        The processed survival function data for time-lapse conditions with the
        following data structure:
            {
                f"{t_tl}": {
                    "time": np.ndarray with the time points,
                    "value": np.ndarray with the survival function values,
                },
                ...
            }

    Examples
    --------
    >>> data = {"50ms": {"time": np.array([0.05, 0.1, 0.15])), "value": np.array([41, 23, 8])},
                "1s": {"time": np.array([1.0, 2.0]), "value": np.array([34, 9])}}
    >>> process_data(data)
    {"0.05s": {"time": np.array([0.1, 0.15]), "value": np.array([1.0, 0.34782609])},
     "1s": {"time": np.array([1.0, 2.0]), "value": np.array([1.0, 0.26470588])}}

    If `delete` is set to False:
    >>> data = {"50ms": {"time": np.array([0.05, 0.1, 0.15]), "value": np.array([41, 23, 8])},
                "1s": {"time": np.array([1.0, 2.0]), "value": np.array([34, 9])}}
    >>> process_data(data, delete=False)
    {"0.05s": {"time": np.array([0.05, 0.1, 0.15]), "value": [1.0, 0.56097561, 0.19512195])},
    "1s": {"time": np.array([1.0, 2.0]), "value": np.array([1.0, 0.26470588])}}
    """

    # Format the data such that all the keys are in seconds
    data_processed = fmt_t_str_data(data)

    if delete:
        t_tls = list(data_processed.keys())
        t_tls.sort()  # Sort the key values from low to high

        t_tl_min = t_tls[0]
        s_min = get_time_sec(t_tl_min)
        if math.isclose(s_min, data_processed[t_tl_min]["time"][0]):
            data_processed[t_tl_min]["time"] = data_processed[t_tl_min]["time"][1:]
            data_processed[t_tl_min]["value"] = data_processed[t_tl_min]["value"][1:]
            print(
                "WARNING: The first data point was deleted! See documentation to",
                "understand why. This is correct behaviour, but important to be aware",
                "of.",
            )

    # Normalize the data
    for t_tl in data_processed.keys():
        data_processed[t_tl]["value"] = (
            data_processed[t_tl]["value"] / data_processed[t_tl]["value"][0]
        )

    return data_processed


def combine_data(
    data1: Dict[str, Dict[str, np.ndarray]],
    data2: Dict[str, Dict[str, np.ndarray]],
    both: bool = True,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Function combines two datasets into one dataset. The data of the different
    time-lapse times is combined to get a new dataset where every time-lapse time key
    has only one time and one value array. The data of time-lapse times will be added up
    if the original two datasets have the same time-lapse time key. However, if both is
    set to True then also time-lapse time keys which are only present in one of the
    datasets are added with its corresponding time and value array.

    Parameters
    ----------
    data1 : Dict[str, Dict[str, np.ndarray]]
        A dictionary mapping keys (time-lapse conditions) to the corresponding time and
        value arrays of the survival functions. For example::

            {
                "0.05s": {
                    "time": array([0.05, 0.1, 0.15, ...]), "value": array([1.000e+04,
                    8.464e+03, 7.396e+03, ...]),
                }, "1s": {
                    "time": array([1., 2., 3., 4., ...]), "value": array([1.000e+04,
                    6.925e+03, 5.541e+03, 4.756e+03, ...]),
                },
            }

    data2 : Dict[str, Dict[str, np.ndarray]]
        A dictionary mapping keys (time-lapse conditions) to the corresponding time and
        value arrays of the survival functions. Same structure as data1.
    both : bool, optional
        Indicates whether only time-lapse times that are present in both datasets
        should be combined and put into the combined dataset, by default True.

    Returns
    -------
    Dict[str, Dict[str, np.ndarray]]
        A dictionary mapping keys (time-lapse conditions) to the corresponding combined
        time and value arrays of the survival functions. For example::

            {
                "0.05s": {
                    "time": array([0.05, 0.1, 0.15, ...]), "value": array([1.000e+04,
                    8.464e+03, 7.396e+03, ...]),
                }, "1s": {
                    "time": array([1., 2., 3., 4., ...]), "value": array([1.000e+04,
                    6.925e+03, 5.541e+03, 4.756e+03, ...]),
                },
            }

    Raises
    ------
    RuntimeWarning
        If something unexpected happens which should not happen.

    Examples
    --------
    >>> data1 = {"0.05s": {...}, "0.2s": {...}, "1s": {...}, "5s": {...}}
    >>> data2 = {"0.05s": {...}, "0.4s": {...}, "1s": {...}, "5s": {...}}
    >>> data_combined_both = combine_data(data1, data2, both = True)
    >>> data_combined_both
    {"0.05s": {...}, "1s": {...}, "5s": {...}}

    >>> data_combined = combine_data(data1, data2, both = False)
    >>> data_combined
    {"0.05s": {...}, "0.2s": {...}, "0.4": {...}, "1s": {...}, "5s": {...}}

    """

    # Format the time-lapse time of both datasets, to make sure they are equally
    # formatted.
    data1 = fmt_t_str_data(data1)
    data2 = fmt_t_str_data(data2)

    # Distinguish between time-lapse time keys that are in both datasets, and those
    # that are not in both datasets
    keys1 = set(data1.keys())
    keys2 = set(data2.keys())

    keys_both = keys1.intersection(keys2)
    keys1_only = keys1.difference(keys_both)
    keys2_only = keys2.difference(keys_both)

    # Convert all the keys to lists and sort them
    keys_both = sorted(list(keys_both))
    keys1_only = sorted(list(keys1_only))
    keys2_only = sorted(list(keys2_only))

    # Create the combined dataset dict and fill it
    data_combined = dict()

    for key in keys_both:

        data_combined[key] = dict()

        time1 = data1[key]["time"]
        time2 = data2[key]["time"]

        dt1 = time1[1] - time1[0]
        dt2 = time2[1] - time2[0]

        if not math.isclose(dt1, dt2):
            raise RuntimeWarning("Something went wrong with dt1 and dt2")

        time1_min = np.amin(time1)
        time1_max = np.amax(time1)

        time2_min = np.amin(time2)
        time2_max = np.amax(time2)

        time_min = min(time1_min, time2_min)
        time_max = max(time1_max, time2_max)

        n = int(round(((time_max - time_min) / dt1) + 1, 0))
        time = np.linspace(time_min, time_max + dt1, num=n + 1, endpoint=True)
        value = np.zeros(time.shape[0])

        idx1_start = int(round(((time1_min - time_min) / dt1), 0))
        idx1_end = int(round(((time1_max - time_max) / dt1), 0)) - 1

        idx2_start = int(round(((time2_min - time_min) / dt2), 0))
        idx2_end = int(round(((time2_max - time_max) / dt2), 0)) - 1

        value[idx1_start:idx1_end] = data1[key]["value"]
        value[idx2_start:idx2_end] += data2[key]["value"]

        time = time[:-1]
        value = value[:-1]

        data_combined[key]["time"] = time
        data_combined[key]["value"] = value

    if not both:
        for key in keys1_only:

            data_combined[key] = dict()

            data_combined[key]["time"] = data1[key]["time"]
            data_combined[key]["value"] = data1[key]["value"]

        for key in keys2_only:

            data_combined[key] = dict()

            data_combined[key]["time"] = data2[key]["time"]
            data_combined[key]["value"] = data2[key]["value"]
    else:
        print(
            "WARNING!! (Set both to True to include all the data)\n"
            f"Data of dataset 1 with time-lapse time {keys1_only} not used.\n"
            f"Data of dataset 2 with time-lapse time {keys2_only} not used."
        )

    return data_combined


def isvalid_parameters_grid(parameters: Dict) -> bool:
    """Function checks whether the provided parameters are valid for GRID fitting.

    Parameters
    ----------
    parameters : Dict
        Dictionary containing all the parameters needed to perform the GRID fitting.

    Returns
    -------
    bool
        Defines whether the dictionary is valid or not.

    Notes
    -----
    Valid key, value pairs:

    "k": np.ndarray,
    "k_min": float,
    "k_max": float,
    "N": int,
    "reg_weight": float,
    "fit_a": bool,
    "a_fixed": None | float

    Examples
    --------
    >>> parameters = {
        "k_min": 10 ** (-3),
        "k_max": 10**1,
        "N": 200,
        "scale": "log",
        "reg_weight": 0.01,
        "fit_a": True,
        "a_fixed": None,
    }
    >>> isvalid_parameters_grid(parameters)
    True

    >>> parameters = {
        "k": [0.001, 0.05, 1.5, 5.6],
        "reg_weight": 0.01,
        "fit_a": True,
        "a_fixed": None,
    }
    >>> isvalid_parameters_grid(parameters)
    True

    """
    KEYS_ALLOWED = set(
        [
            "k",
            "k_min",
            "k_max",
            "N",
            "reg_weight",
            "scale",
            "fit_a",
            "a_fixed",
        ]
    )

    param_keys = set(parameters.keys())

    if len(param_keys.difference(KEYS_ALLOWED)) != 0:
        raise ValueError(
            f"The keywords: {param_keys.difference(KEYS_ALLOWED)} are not valid \
                parameters t_tls"
        )
    elif "k" not in parameters and (
        "k_min" not in parameters
        or "k_max" not in parameters
        or "N" not in parameters
        or "scale" not in parameters
    ):
        if "k_min" not in parameters:
            raise ValueError("Key 'k_min' is missing in the parameters variable")
        elif "k_max" not in parameters:
            raise ValueError("Key 'k_max' is missing in the parameters variable")
        elif "N" not in parameters:
            raise ValueError("Key 'N' is missing in the parameters variable")
        elif "scale" not in parameters:
            raise ValueError("Key 'scale' is missing in the parameters variable")

        # Check if the 'scale' parameter is a valid value
        if parameters["scale"] not in set(["log", "linear"]):
            raise ValueError("Value for key 'scale' should be either log or linear")

    elif "reg_weight" not in parameters:
        raise ValueError("Key 'reg_weight' is missing in the parameters variable")
    elif "fit_a" not in parameters:
        raise ValueError("Key 'fit_a' is missing in the parameters variable")

    if not parameters["fit_a"] and "a_fixed" not in parameters:
        raise ValueError(
            "If the photobleaching number should not be fitted then a photobleaching \
                number (a) should be provided"
        )
    return True


def isvalid_parameters_n_exp(parameters: Dict) -> bool:
    """Function checks whether the provided parameters are valid for n-exp fitting.

    Parameters
    ----------
    parameters : Dict
        Dictionary containing all the parameters needed to perform the n-exp fitting.

    Returns
    -------
    bool
        Defines whether the dictionary is valid or not.

    Notes
    -----
    Valid key, value pairs:
    "n_exp": int | Sequence
    "k": np.ndarray
    "k_min": float
    "k_max": float
    "fit_a": bool
    "a_fixed": None | float

    Examples
    --------
    >>> parameters = {
            "n_exp": 3,
            "k_min": 10**(-3),
            "k_max": 10**1,
            "fit_a": True,
            "a_fixed": None,
        }
    >>> isvalid_parameters_n_exp(parameters)
    True

    >>> parameters = {
        "k": [0.001, 0.05, 1.5, 5.6],
        "fit_a": True,
        "a_fixed": None,
    }
    >>> isvalid_parameters_n_exp(parameters)
    True
    """
    KEYS_ALLOWED = set(
        [
            "n_exp",
            "k",
            "k_min",
            "k_max",
            "fit_a",
            "a_fixed",
        ]
    )

    param_keys = set(parameters.keys())

    if len(param_keys.difference(KEYS_ALLOWED)) != 0:
        raise ValueError(
            f"The keywords: {param_keys.difference(KEYS_ALLOWED)} are not valid \
                parameters t_tls"
        )

    elif "k" not in parameters and (
        "k_min" not in parameters
        or "k_max" not in parameters
        or "n_exp" not in parameters
    ):
        if "k_min" not in parameters:
            raise ValueError("Key 'k_min' is missing in the parameters variable")
        elif "k_max" not in parameters:
            raise ValueError("Key 'k_max' is missing in the parameters variable")
        elif "n_exp" not in parameters:
            raise ValueError("Key 'n_exp' is missing in the parameters variable")
    elif "fit_a" not in parameters:
        raise ValueError("Key 'fit_a' is missing in the parameters variable")
    if not parameters["fit_a"] and "a_fixed" not in parameters:
        raise ValueError(
            "If the photobleaching number should not be fitted then a photobleaching \
                number (a) should be provided"
        )

    return True


def split_parameters(parameters: Dict) -> Dict:
    """Function splits a provided parameters dictionary into two different
    dictionaries. One dictionary for parameters required for GRID fitting and one
    dictionary for parameters required for n-exp fitting."""
    KEYS_GRID = set(
        [
            "k",
            "k_min",
            "k_max",
            "N",
            "reg_weight",
            "scale",
            "fit_a",
            "a_fixed",
        ]
    )

    KEYS_N_EXP = set(
        [
            "n_exp",
            "k",
            "k_min",
            "k_max",
            "fit_a",
            "a_fixed",
        ]
    )

    parameters_grid = dict()
    parameters_n_exp = dict()

    for key, value in parameters.items():

        if key in KEYS_GRID:
            parameters_grid[key] = value
        if key in KEYS_N_EXP:
            parameters_n_exp[key] = value

    return parameters_grid, parameters_n_exp
