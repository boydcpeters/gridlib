"""
Module with utility functions required for the plotting.
"""
from typing import Dict
from .. import data_utils


def _fmt_t_str_plot(t_str):
    """Function returns a formatted string which can be plotted inside figures."""
    t_s, d_places = data_utils.get_time_sec(t_str, return_num_decimals=True)
    return rf"${t_s:.{d_places}f}\,\mathrm{{s}}$"


def _get_key_to_value_i(i: int, key_to_value: Dict):
    """Function gets the key_to_value at position i.
    
    Parameters
    ----------
    i: int
        Index from which to take the key_to_value.
    key_to_value: Dict
        Dictionary with keys and values, which can either be sequences or a single value
    
    Example
    -------
    >>> key_to_value = {"color": ["r", "b", "g"], "linestyle": ["-", "--", "-."],
                        "linewidth": 1}
    >>> _get_key_to_value_i(1, key_to_value)
    {"color": "b", "linestyle": "--", "linewidth": 1}
    """

    # Check the length of the values
    # They can be either 1 or one other value, for example either 1 or 3, but it cannot
    # be 1, 3 and 5
    value_len = set()
    for key, value in key_to_value:
        value_len.add(len(value))
    
    if len(value_len) > 2:
        raise ValueError(f"`key_to_value` is not valid, it contains {len(value_len)} " \
                            "different sequence lengths.")


    key_to_value_i = dict()
    for key, value in key_to_value.items():
        if len(value) > 1:
            key_to_value_i[key] = value[i]
        else:
            key_to_value_i[key] = value
    
    return key_to_value_i