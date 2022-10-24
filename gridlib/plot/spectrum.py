"""
Module with functions to plot event spectrum and state spectrum.
"""
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredText

from . import _plot_utils


def _base_spectrum(
    key_to_k_and_weight,
    scale: str = "log",
    threshold: float = 0.0,
    xlim: Tuple[float, float] = None,
    ylim: Tuple[float, float] = None,
    figsize: Tuple[float, float] = (8, 5),
    color=None,
    add_legend: bool = True,
):
    """Base function that plots the spectra in one figure.

    Parameters
    ----------
    key_to_k_and_weight : Dict[str, Dict[str, np.ndarray]]
        The decay rates and weights wrapped in the following structure:
            {
                "label": {
                    "k": np.ndarray with the decay rates,
                    "weight": np.ndarray with the respective weights
                }
            }
    scale : {"log", "linear"}, optional
        The scale on the x-axis. The default value is "log".
    threshold : float, optional
        The weight threshold for plotting lines, by default 0.0.
    xlim : Tuple[float, float], optional
        A tuple of the x-axis limits, by default None.
    ylim : Tuple[float, float], optional
        A tuple of the y-axis limits, by default None.
    figsize : Tuple[float, float], optional
        A tuple of the figure size, by default (8, 5).
    color : color or sequence of colors, optional
        The color or colors to use for the plotting. The standard gridlib colors are
        used if the value is set to None. The value is by default None.
    add_legend : bool, optional
        Indicates whether the legend needs to be plotted, by default True.

    Returns
    -------
    fig : matplotlib.Figure
        TODO: link this to matplotlib.Figure documentation
        https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure
    ax: matplotlib.axes.Axes
        TODO: link to the matplotlib.axes.Axes documentation
        https://matplotlib.org/stable/api/axes_api.html#matplotlib.axes.Axes
    """

    # Check if the scale option is valid
    if scale not in ["log", "linear"]:
        raise ValueError("The variable 'scale' does not contain a valid value.")

    # Colors
    gridlib_colors = _plot_utils._gridlib_colors()
    if color is None and len(key_to_k_and_weight) <= len(gridlib_colors):
        color = gridlib_colors[:]
    elif color is None:
        color = _plot_utils._default_colors()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    # Retrieve the keys and sort them
    keys = list(key_to_k_and_weight.keys())
    keys.sort()
    if "grid" in keys:
        # First remove it
        keys.remove("grid")
        # and place it at the start of the list so it is plotted first
        keys.insert(0, "grid")

    for i, key in enumerate(keys):
        k = key_to_k_and_weight[key]["k"]
        weight = key_to_k_and_weight[key]["weight"]

        if key == "grid":
            linestyle_vlines = "solid"
            label = "GRID spectrum"
        else:
            linestyle_vlines = "dashed"
            label = key

        idx = weight >= threshold
        ax.plot(
            k[idx],
            weight[idx],
            linestyle="None",
            marker="o",
            markersize=3,
            color=color[i],
            label=label,
        )
        ax.vlines(
            k[idx],
            np.zeros(k.shape[0])[idx],
            weight[idx],
            linewidth=1,
            linestyles=linestyle_vlines,
            color=color[i],
            label=label,
        )

    if scale == "log":
        ax.set_xscale("log")

    # Axis limits
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    if add_legend:
        # Legend
        # Remove duplicate labels
        # Implementation from: https://stackoverflow.com/questions/13588920/stop-matplotlib-repeating-labels-in-legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())  # , loc="lower left")

    return fig, ax


# TODO: think about setting the weight for a single-exponential to 0.2 for visual reasons


def event_spectrum(
    fit_values,
    scale: str = "log",
    threshold: float = 0.0,
    xlim: Tuple[float, float] = None,
    ylim: Tuple[float, float] = None,
    figsize: Tuple[float, float] = (8, 5),
    color=None,
    add_legend: bool = True,
):
    """Function plots the event spectrum of different fit values in one figure.

    Parameters
    ----------
    key_to_k_and_weight : Dict[str, Dict[str, np.ndarray]]
        The decay rates and weights wrapped in the following structure:
            {
                "label": {
                    "k": np.ndarray with the decay rates,
                    "weight": np.ndarray with the respective weights
                }
            }
    scale : {"log", "linear"}, optional
        The scale on the x-axis. The default value is "log".
    threshold : float, optional
        The weight threshold for plotting lines, by default 0.0.
    xlim : Tuple[float, float], optional
        A tuple of the x-axis limits, by default None.
    ylim : Tuple[float, float], optional
        A tuple of the y-axis limits, by default None.
    figsize : Tuple[float, float], optional
        A tuple of the figure size, by default (8, 5).
    color : color or sequence of colors, optional
        The color or colors to use for the plotting. The standard gridlib colors are
        used if the value is set to None. The value is by default None.
    add_legend : bool, optional
        Indicates whether the legend needs to be plotted, by default True.

    Returns
    -------
    fig : matplotlib.Figure
        TODO: link this to matplotlib.Figure documentation
        https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure
    ax: matplotlib.axes.Axes
        TODO: link to the matplotlib.axes.Axes documentation
        https://matplotlib.org/stable/api/axes_api.html#matplotlib.axes.Axes
    """

    a = None  # Photobleaching number

    key_to_k_and_weight = dict()
    for key in fit_values.keys():

        k = fit_values[key]["k"]
        s = fit_values[key]["s"]

        key_to_k_and_weight[key] = dict()
        key_to_k_and_weight[key]["k"] = k
        key_to_k_and_weight[key]["weight"] = s

        # Store the photobleaching number if it is in the fit results
        if key == "grid":
            a = fit_values[key]["a"]

    fig, ax = _base_spectrum(
        key_to_k_and_weight,
        scale=scale,
        threshold=threshold,
        xlim=xlim,
        ylim=ylim,
        figsize=figsize,
        color=color,
        add_legend=add_legend,
    )

    # Labels
    ax.set_xlabel("dissociation rate (1/s)")
    ax.set_ylabel("event spectrum")

    if a is not None:
        # add the bleaching number in the plot if there were grid fit values provided
        # https://stackoverflow.com/questions/23112903/matplotlib-text-boxes-automatic-position
        anchored_text = AnchoredText(f"a = {a:.5f}", loc="center left", frameon=False)
        ax.add_artist(anchored_text)

    return fig, ax


def state_spectrum(
    fit_values,
    scale: str = "log",
    threshold: float = 0.0,
    xlim: Tuple[float, float] = None,
    ylim: Tuple[float, float] = None,
    figsize: Tuple[float, float] = (8, 5),
    color=None,
    add_legend: bool = True,
):
    """Function plots the state spectrum of different fit values in one figure.

    Parameters
    ----------
    key_to_k_and_weight : Dict[str, Dict[str, np.ndarray]]
        The decay rates and weights wrapped in the following structure:
            {
                "label": {
                    "k": np.ndarray with the decay rates,
                    "weight": np.ndarray with the respective weights
                }
            }
    scale : {"log", "linear"}, optional
        The scale on the x-axis. The default value is "log".
    threshold : float, optional
        The weight threshold for plotting lines, by default 0.0.
    xlim : Tuple[float, float], optional
        A tuple of the x-axis limits, by default None.
    ylim : Tuple[float, float], optional
        A tuple of the y-axis limits, by default None.
    figsize : Tuple[float, float], optional
        A tuple of the figure size, by default (8, 5).
    color : color or sequence of colors, optional
        The color or colors to use for the plotting. The standard gridlib colors are
        used if the value is set to None. The value is by default None.
    add_legend : bool, optional
        Indicates whether the legend needs to be plotted, by default True.

    Returns
    -------
    fig : matplotlib.Figure
        TODO: link this to matplotlib.Figure documentation
        https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure
    ax: matplotlib.axes.Axes
        TODO: link to the matplotlib.axes.Axes documentation
        https://matplotlib.org/stable/api/axes_api.html#matplotlib.axes.Axes
    """

    a = None  # Photobleaching number

    key_to_k_and_weight = dict()
    for key in fit_values.keys():

        k = fit_values[key]["k"]
        s = fit_values[key]["s"]

        s_state = (1 / k) * s
        s_state = s_state / np.sum(s_state)  # normalization

        key_to_k_and_weight[key] = dict()
        key_to_k_and_weight[key]["k"] = k
        key_to_k_and_weight[key]["weight"] = s_state

        # Store the photobleaching number if it is in the fit results
        if key == "grid":
            a = fit_values[key]["a"]

    fig, ax = _base_spectrum(
        key_to_k_and_weight,
        scale=scale,
        threshold=threshold,
        xlim=xlim,
        ylim=ylim,
        figsize=figsize,
        color=color,
        add_legend=add_legend,
    )

    # Labels
    ax.set_xlabel("dissociation rate (1/s)")
    ax.set_ylabel("state spectrum")

    if a is not None:
        # add the bleaching number in the plot if there were grid fit values provided
        # https://stackoverflow.com/questions/23112903/matplotlib-text-boxes-automatic-position
        anchored_text = AnchoredText(f"a = {a:.5f}", loc="center right", frameon=False)
        ax.add_artist(anchored_text)

    return fig, ax
