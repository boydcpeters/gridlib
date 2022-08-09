# Here are utils function, such as for checking the whether the parameters are
#  valid
# and for reformatting the t_tls arguments
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
    >>> num_decimal_places("5.32")
    2
    >>> num_decimal_places("2.")
    0
    >>> num_decimal_places("98.8572")
    4
    >>> num_decimal_places("3.4120")
    3
    >>> num_decimal_places("11")
    0
    >>> num_decimal_places("0.05")
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
                "t_tl": {
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
    >>> data = {"50ms": {"time": array([0.05, 0.1, 0.15]), "value": array([41, 23, 8])},
                "1s": {"time": array([1.0, 2.0]), "value": array([34, 9])}}
    >>> fmt_t_str_data(data)
    {"0.05s": {"time": array([0.05, 0.1, 0.15]), "value": array([41, 23, 8])},
     "1s": {"time": array([1.0, 2.0]), "value": array([34, 9])}}
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
                "t_tl": {
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
                "t_tl": {
                    "time": np.ndarray with the time points,
                    "value": np.ndarray with the survival function values,
                },
                ...
            }

    Examples
    --------
    >>> data = {"50ms": {"time": array([0.05, 0.1, 0.15])), "value": array([41, 23, 8])},
                "1s": {"time": array([1.0, 2.0]), "value": array([34, 9])}}
    >>> process_data(data)
    {"0.05s": {"time": array([0.1, 0.15]), "value": array([1.0, 0.34782609])},
     "1s": {"time": array([1.0, 2.0]), "value": array([1.0, 0.26470588])}}

    If `delete` is set to False:
    >>> data = {"50ms": {"time": array([0.05, 0.1, 0.15]), "value": array([41, 23, 8])},
                "1s": {"time": array([1.0, 2.0]), "value": array([34, 9])}}
    >>> process_data(data, delete=False)
    {"0.05s": {"time": array([0.05, 0.1, 0.15]), "value": [1.0, 0.56097561, 0.19512195])},
    "1s": {"time": array([1.0, 2.0]), "value": array([1.0, 0.26470588])}}
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
                "WARNING: The first data point was deleted! See documentation to ",
                "understand why. This is correct behaviour, but important to be aware ",
                "of.",
            )

    # Normalize the data
    for t_tl in data_processed.keys():
        data_processed[t_tl]["value"] = (
            data_processed[t_tl]["value"] / data_processed[t_tl]["value"][0]
        )

    return data_processed


def isvalid_parameters(parameters: Dict) -> bool:
    """Function checks whether the provided parameters are valid.

    Parameters
    ----------
    parameters : Dict
        Dictionary containing all the parameters needed to perform the GRID fitting.

    Returns
    -------
    bool
        Defines whether the dictionary is valid or not.

    Example
    -------
    A valid parameters dictionary:

    parameters = {
        "t_int": 0.05,
        "k_min": 10**(-3),
        "k_max": 10**1,
        "N": 200,
        "scale": "log",  # "linear"
        "reg_weight": 0.01,
        "fit_a": True,
        "fit_a_exp": True,
        "a_fixed": None
    }
    """
    KEYS_ALLOWED = set(
        [
            "t_int",
            "k_min",
            "k_max",
            "N",
            "reg_weight",
            "scale",
            "fit_a",
            "fit_a_exp",
            "a_fixed",
        ]
    )

    t_tls_para = set(parameters.keys())

    if len(t_tls_para.difference(KEYS_ALLOWED)) != 0:
        raise ValueError(
            f"The keywords: {t_tls_para.difference(KEYS_ALLOWED)} are not valid \
                parameters t_tls"
        )

    if "t_int" not in parameters:
        raise ValueError("Key 't_int' is missing in the parameters variable")
    elif "k_min" not in parameters:
        raise ValueError("Key 'k_min' is missing in the parameters variable")
    elif "k_max" not in parameters:
        raise ValueError("Key 'k_max' is missing in the parameters variable")
    elif "N" not in parameters:
        raise ValueError("Key 'N' is missing in the parameters variable")
    elif "scale" not in parameters:
        raise ValueError("Key 'scale' is missing in the parameters variable")
    elif "reg_weight" not in parameters:
        raise ValueError("Key 'reg_weight' is missing in the parameters variable")
    elif "fit_a" not in parameters:
        raise ValueError("Key 'fit_a' is missing in the parameters variable")
    elif "fit_a_exp" not in parameters:
        raise ValueError("Key 'fit_a_exp' is missing in the parameters variable")

    if parameters["scale"] not in set(["log", "linear"]):
        raise ValueError("Value for key 'scale' should be either log or linear")
    if not parameters["fit_a"] and "a_fixed" not in parameters:
        raise ValueError(
            "If the photobleaching number should not be fitted then a photobleaching \
                number (a) should be provided"
        )
    if not parameters["fit_a"] and parameters["fit_a_exp"]:
        print(
            "WARNING! Photobleaching number is fixed for GRID but is still set to be \
            fitted for the multi-exponentials!"
        )

    return True
