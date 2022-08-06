"""
Module with utility functions required for the plotting.
"""

from .. import data_utils


def _fmt_t_str_plot(t_str):
    """Function returns a formatted string which can be plotted inside figures."""
    t_s, d_places = data_utils.get_time_sec(t_str, return_num_decimals=True)
    return rf"${t_s:.{d_places}f}\,\mathrm{{s}}$"
