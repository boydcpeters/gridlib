"""
Module with functions to plot survival functions.
"""
import enum
from typing import Tuple, Dict, Union
import pathlib
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.lines import Line2D

from .. import data_utils
from . import _plot_utils
from .. import compute

# TODO: UPDATE DOCSTRING
def base_data_sf(data, process_data_flag: bool = True, figsize: Tuple[float, float] = (10, 8), **kwargs):
    """Function plots """
    
    # Check if it is a single dict or a sequence
    # if it is a single dict, make it a sequence so it works with the rest of the
    # function
    if isinstance(data, dict):
        data = [data]

    if process_data_flag:
        # Process data
        for i in range(len(data)):
            data[i] = data_utils.process_data(data[i])

    fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    text_points = set()

    # Sort the time-lapse conditions from low to high, required for proper
    # text positions
    for i, d in enumerate(data):

        kwargs_d = _plot_utils._get_key_to_value_i(i, kwargs)

        for t_tl in sorted(data[i].keys()):
            time = d[t_tl]["time"]
            value = d[t_tl]["value"]

            ax1.loglog(time, value, kwargs_d)

            if i == 0:
                # Create the new point position for the text
                text_point = (time[0] * 0.6, value[0] * 1.1)

                # Check if the new text will not overlap
                if text_point in text_points:
                    text_point = (time[0] * 1.3, value[0] * 1.1)

                # This is a hack-job and should not be a long term fix
                # TODO: fix this do it automatically
                a = _plot_utils._fmt_t_str_plot(t_tl)
                if a == "$0.39\,\mathrm{s}$":
                    text_point = (time[0] * 1.1, value[0] * 1.1)

                ax1.text(
                    text_point[0], text_point[1], _plot_utils._fmt_t_str_plot(t_tl)
                )
                text_points.add(text_point)

    return fig1, ax1

# TODO: UPDATE DOCSTRING

def data_vs_multi_exp(
    data: Dict[str, Dict[str, np.ndarray]],
    fit_values_multi_exp: Dict[str, Dict[str, np.ndarray]],
    process_data_flag: bool = True,
    xlim: Tuple[float, float] = None,
    ylim: Tuple[float, float] = None,
    figsize: Tuple[float, float] = (10, 8),
    path_save: Union[str, pathlib.Path] = None,
    **kwargs
):
    """Function plots the survival function of the true data and the multi-exponential curves.

    Parameters
    ----------
    data: Dict[str, Dict[str, np.ndarray]]
        Data of the survival function from real data with the following data structure:
        {
            f"{t_tl}": {
                "time": np.ndarray with all the time values,
                "value": np.ndarray with all the survival function values corresponding to the
                    respective time value
            }
        }
    data_multi_exp: Dict[str, Dict[str, np.ndarray]]
        Data of the computed GRID survival functions. The dictionary structure is as follows:
        {
            f"{t_tl}": {
                "time": np.ndarray with all the time values,
                "value": np.ndarray with all the survival function values corresponding to the
                    respective time value
            }
        }
    """

    # Create a Path() from path_save if not None and it is a str
    if path_save is not None and isinstance(path_save, str):
        path_save = pathlib.Path(path_save)

    key = fit_values_multi_exp.keys()[0]
    n_exp = fit_values_multi_exp[key]["k"].shape[0]

    if kwargs is None:
        # Set the default settings
        kwargs = dict()
        # color: data, data_multi_exp
        kwargs["label"] = ["data", f"{n_exp}-exp"]
        kwargs["color"] = ["#007972", "#fe9901"]
        kwargs["linewidth"] = 1
        kwargs["linestyle"] = ["solid", "dashed"]


    data_multi_exp = compute.compute_multi_exp(fit_values_multi_exp, data)

    fig1, ax1 = base_data_sf([data, data_multi_exp], process_data_flag=process_data_flag, figsize=figsize, kwargs=kwargs)

    # Legend
    ax1.legend(loc="lower left")

    # Axis limits
    if xlim is not None:
        ax1.set_xlim(xlim)
    if ylim is not None:
        ax1.set_ylim(ylim)

    # Labels
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("survival function")

    if path_save is not None:
        fig1.savefig(path_save, bbox_inches="tight", dpi=200)
        plt.close(fig1)
    
    return fig1, ax1

# TODO: UPDATE DOCSTRING

def data_vs_grid(
    data: Dict[str, Dict[str, np.ndarray]],
    fit_values_grid: Dict[str, Dict[str, np.ndarray]],
    process_data_flag: bool = True,
    xlim: Tuple[float, float] = None,
    ylim: Tuple[float, float] = None,
    figsize: Tuple[float, float] = (10, 8),
    path_save: Union[str, pathlib.Path] = None,
    **kwargs
    ):
    """Function plots the survival function of the true data and the GRID curves.

    Parameters
    ----------
    data: Dict[str, Dict[str, np.ndarray]]
        Data of the survival function from real data with the following data structure:
        {
            f"{t_tl}": {
                "time": np.ndarray with all the time values,
                "value": np.ndarray with all the survival function values corresponding to the
                    respective time value
            }
        }
    data_grid: Dict[str, Dict[str, np.ndarray]]
        Data of the computed GRID survival functions. The dictionary structure is as follows:
        {
            f"{t_tl}": {
                "time": np.ndarray with all the time values,
                "value": np.ndarray with all the survival function values corresponding to the
                    respective time value
            }
        }
    xlim: Tuple[float, float] = None, ylim: Tuple[float, float] = None, path_save: str or pathlib.Path
        Path designates the place where the figure should be saved if a value is set. The figure is
        not saved if the values is set to None. (default None)

    Returns
    -------
    None
    """
    # Create a Path() from path_save if not None and it is a str
    if path_save is not None and isinstance(path_save, str):
        path_save = pathlib.Path(path_save)

    if kwargs is None:
        # Set the default settings
        kwargs = dict()
        # color: data, data_multi_exp
        kwargs["label"] = ["data", "GRID"]
        kwargs["color"] = ["#007972", "#fe9901"]
        kwargs["linewidth"] = 1
        kwargs["linestyle"] = ["solid", "dashed"]

    data_grid = compute.compute_grid_curves(fit_values_grid, data)

    fig1, ax1 = base_data_sf([data, data_grid], process_data_flag=process_data_flag, figsize=figsize, kwargs=kwargs)

    # Legend
    ax1.legend(loc="lower left")

    # Axis limits
    if xlim is not None:
        ax1.set_xlim(xlim)
    if ylim is not None:
        ax1.set_ylim(ylim)

    # Labels
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("survival function")

    if path_save is not None:
        fig1.savefig(path_save, bbox_inches="tight", dpi=200)
        plt.close(fig1)
    
    return fig1, ax1