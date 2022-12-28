"""
Module with functions to plot survival functions.
"""

import pathlib
from typing import Dict, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from .. import compute, data_utils
from . import _plot_utils


# TODO: UPDATE DOCSTRING
def _base_data_multiple(
    data,
    process_data_flag: bool = True,
    xlim: Tuple[float, float] = None,
    ylim: Tuple[float, float] = None,
    figsize: Tuple[float, float] = (6, 4),
    kwargs_plot: Dict = None,
    kwargs_text: Dict = None,
):
    """_summary_

    Parameters
    ----------
    data : _type_
        _description_
    process_data_flag : bool, optional
        _description_, by default True
    xlim : Tuple[float, float], optional
        _description_, by default None
    ylim : Tuple[float, float], optional
        _description_, by default None
    figsize : Tuple[float, float], optional
        _description_, by default (6, 4)
    kwargs_plot : Dict, optional
        _description_, by default None
    kwargs_text : Dict, optional
        _description_, by default None

    Returns
    -------
    fig : matplotlib.Figure
        TODO: link this to matplotlib.Figure documentation
        https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure
    ax: matplotlib.axes.Axes
        TODO: link to the matplotlib.axes.Axes documentation
        https://matplotlib.org/stable/api/axes_api.html#matplotlib.axes.Axes
    """

    # Create empty dictionaries for the kwargs_... if they are None.
    # Mutable default values leads to dangerous behaviour
    # (https://stackoverflow.com/questions/1132941/least-astonishment-and-the-mutable-default-argument)
    # Hence, it is done this way.

    if kwargs_plot is None:
        kwargs_plot = dict()
    if kwargs_text is None:
        kwargs_text = dict()

    # Check if it is a single dict or a sequence
    # if it is a single dict, make it a sequence so it works with the rest of the
    # function
    if isinstance(data, dict):
        data = [data]

    if process_data_flag:
        # Process data
        for i in range(len(data)):
            data[i] = data_utils.process_data(data[i])

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    text_points = set()

    # Sort the time-lapse conditions from low to high, required for proper
    # text positions
    for i, d in enumerate(data):

        # Plotting kwargs
        kwargs_d = _plot_utils._get_key_to_value_i(i, kwargs_plot)

        for t_tl in sorted(data[i].keys()):
            time = d[t_tl]["time"]
            value = d[t_tl]["value"]

            ax.loglog(time, value, **kwargs_d)

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

                ax.text(
                    text_point[0],
                    text_point[1],
                    _plot_utils._fmt_t_str_plot(t_tl),
                    **kwargs_text,
                )
                text_points.add(text_point)

    # Axis limits
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Labels
    ax.set_xlabel("time (s)")
    ax.set_ylabel("survival function")

    # Legend
    # Remove duplicate labels
    # Implementation from: https://stackoverflow.com/questions/13588920/stop-matplotlib-repeating-labels-in-legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())  # , loc="lower left")

    return fig, ax


def data_multiple(
    data: Dict[str, Dict[str, np.ndarray]],
    process_data_flag: bool = True,
    xlim: Tuple[float, float] = None,
    ylim: Tuple[float, float] = None,
    figsize: Tuple[float, float] = (6, 4),
    kwargs_plot: Dict = None,
    kwargs_text: Dict = None,
):
    """Function plots a single or multiple data dictionaries"""

    # Create empty dictionaries for the kwargs_... if they are None.
    # Mutable default values leads to dangerous behaviour
    # (https://stackoverflow.com/questions/1132941/least-astonishment-and-the-mutable-default-argument)
    # Hence, it is done this way.

    if kwargs_plot is None:
        kwargs_plot = dict()
    if kwargs_text is None:
        kwargs_text = dict()

    # Set the default settings if they are not provided
    if "label" not in kwargs_plot:
        kwargs_plot["label"] = ["data1", "data2"]
    if "color" not in kwargs_plot:
        kwargs_plot["color"] = ["#007972", "#fe9901"]
    if "linewidth" not in kwargs_plot:
        kwargs_plot["linewidth"] = 1
    if "linestyle" not in kwargs_plot:
        kwargs_plot["linestyle"] = ["solid", "dashed"]

    fig, ax = _base_data_multiple(
        data,
        process_data_flag=process_data_flag,
        xlim=xlim,
        ylim=ylim,
        figsize=figsize,
        kwargs_plot=kwargs_plot,
        kwargs_text=kwargs_text,
    )

    return fig, ax


# TODO: fix bug in compute_multi_exp call
# TODO: UPDATE DOCSTRING
# TODO: check how .plot() function shows kwargs, it puts it under "Other parameters"
# although we know have it as a keyword argument with a default value, so we should
# likely just put it under "Parameters"
def data_vs_multi_exp(
    data: Dict[str, Dict[str, np.ndarray]],
    fit_values_multi_exp: Dict[str, Dict[str, np.ndarray]],
    process_data_flag: bool = True,
    xlim: Tuple[float, float] = None,
    ylim: Tuple[float, float] = None,
    figsize: Tuple[float, float] = (6, 4),
    kwargs_plot: Dict = None,
    kwargs_text: Dict = None,
) -> Tuple:
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
        Survival function data for time-lapse conditions with the following data
        structure:
            {
                f"{t_tl}": {
                    "time": np.ndarray with the time points,
                    "value": np.ndarray with the survival function values,
                },
                ...
            }
    """

    # Create empty dictionaries for the kwargs_... if they are None.
    # Mutable default values leads to dangerous behaviour
    # (https://stackoverflow.com/questions/1132941/least-astonishment-and-the-mutable-default-argument)
    # Hence, it is done this way.

    if kwargs_plot is None:
        kwargs_plot = dict()
    if kwargs_text is None:
        kwargs_text = dict()

    n_exp_key = fit_values_multi_exp.keys()[0]
    n_exp = fit_values_multi_exp[n_exp_key]["k"].shape[0]

    # Set the default settings if they are not provided
    if "label" not in kwargs_plot:
        kwargs_plot["label"] = ["data", f"{n_exp}-exp"]
    if "color" not in kwargs_plot:
        kwargs_plot["color"] = ["#007972", "#fe9901"]
    if "linewidth" not in kwargs_plot:
        kwargs_plot["linewidth"] = 1
    if "linestyle" not in kwargs_plot:
        kwargs_plot["linestyle"] = ["solid", "dashed"]

    # k = fit_values_multi_exp[n_exp_key]["k"]
    # s = fit_values_multi_exp[n_exp_key]["s"]
    # a = fit_values_multi_exp[n_exp_key]["a"]

    data_multi_exp = compute.compute_multi_exp_for_data(
        fit_values_multi_exp[n_exp_key], data
    )

    fig, ax = _base_data_multiple(
        [data, data_multi_exp],
        process_data_flag=process_data_flag,
        xlim=xlim,
        ylim=ylim,
        figsize=figsize,
        kwargs_plot=kwargs_plot,
        kwargs_text=kwargs_text,
    )

    return fig, ax


# TODO: UPDATE DOCSTRING


def data_vs_grid(
    data: Dict[str, Dict[str, np.ndarray]],
    fit_values_grid: Dict[str, Dict[str, np.ndarray]],
    process_data_flag: bool = True,
    xlim: Tuple[float, float] = None,
    ylim: Tuple[float, float] = None,
    figsize: Tuple[float, float] = (6, 4),
    kwargs_plot: Dict = None,
    kwargs_text: Dict = None,
):
    """Function plots the survival function of the true data and the GRID curves.

    Parameters
    ----------
    data: Dict[str, Dict[str, np.ndarray]]
        Survival function data for every time-lapse condition with the following data
        structure:
            {
                f"{t_tl}": {
                    "time": np.ndarray with the time points,
                    "value": np.ndarray with the survival function values,
                }
            }
    xlim: Tuple[float, float] = None, ylim: Tuple[float, float] = None, path_save: str or pathlib.Path
        Path designates the place where the figure should be saved if a value is set. The figure is
        not saved if the values is set to None. (default None)

    Returns
    -------
    None
    """

    # Create empty dictionaries for the kwargs_... if they are None.
    # Mutable default values leads to dangerous behaviour
    # (https://stackoverflow.com/questions/1132941/least-astonishment-and-the-mutable-default-argument)
    # Hence, it is done this way.

    if kwargs_plot is None:
        kwargs_plot = dict()
    if kwargs_text is None:
        kwargs_text = dict()

    # Set the default settings if they are not provided
    if "label" not in kwargs_plot:
        kwargs_plot["label"] = ["data", "GRID"]
    if "color" not in kwargs_plot:
        kwargs_plot["color"] = ["#007972", "#fe9901"]
    if "linewidth" not in kwargs_plot:
        kwargs_plot["linewidth"] = 1
    if "linestyle" not in kwargs_plot:
        kwargs_plot["linestyle"] = ["solid", "dashed"]

    # k = fit_values_grid["grid"]["k"]
    # s = fit_values_grid["grid"]["s"]
    # a = fit_values_grid["grid"]["a"]

    data_grid = compute.compute_grid_curves_for_data(fit_values_grid["grid"], data)

    fig, ax = _base_data_multiple(
        [data, data_grid],
        process_data_flag=process_data_flag,
        xlim=xlim,
        ylim=ylim,
        figsize=figsize,
        kwargs_plot=kwargs_plot,
        kwargs_text=kwargs_text,
    )

    return fig, ax
