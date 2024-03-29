"""
Module with functions to plot event heatmap and state heatmap.
"""
from typing import Dict, List, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable

from . import _plot_utils

"""weights can be either event weights or state weights, arrays should be flattened"""

# TODO: if a spectral value is obtained less than two times, it should be omitted according
# to the original paper
def _base_heatmap(
    k_full: np.ndarray,
    weight_full: np.ndarray,
    k_resampled: np.ndarray,
    weight_resampled: np.ndarray,
    scale: str = "log",
    threshold: float = 10e-6,
    xlim: Tuple[float, float] = None,
    ylim: Tuple[float, float] = None,
    figsize: Tuple[float, float] = (6, 4),
    cm_max: int = 20,
    cm_step: int = 2,
    add_legend: bool = True,
):
    """
    Function creates the base heatmap plot required for resampling results plotting.

    Parameters
    ----------
    k_full : np.ndarray
        Decay rates for the full fit results.
    weight_full : np.ndarray
        Amplitudes for the corresponding decay rates. The amplitudes can be either
        event weights or state weights.
    k_resampled : np.ndarray
        decay rates for the resampled fit results. The array should be flattened, so
        it is a 1D array.
    weight_resampled : np.ndarray
        Amplitudes for the corresponding resampled decay rates. The amplitudes can be
        either event weights or state weights. The array should be flattened, so
        it is a 1D array.
    scale : {"log", "linear"}, optional
        The scale of the x-axis. If scale is set to "log" than the x-axis will be
        logarithmic. If scale is set to "linear", the x-axis will be linear, by default
        "log".
    threshold : float, optional
        Minimum weight value that is shown for full data spectrum, by default 10e-6.
    xlim : Tuple[float, float], optional
        A tuple setting the x-axis limits. If the value is set to None, there are no
        limits, by default None.
    ylim : Tuple[float, float], optional
        A tuple setting the y-axis limits. If the value is set to None, there are no
        limits, by default None.
    figsize : Tuple[float, float], optional
        Width, height of the figure in inches, default is (6, 4).
    cm_max : int, optional
        Maximum value of the colormap/colorbar, by default 20.
    cm_step : int, optional
        Step size of the colormap, by default 2.
    add_legend : bool, optional
        If True, a legend is added to the figure, by default True.

    Returns
    -------
    fig: :py:class:`matplotlib.figure.Figure`
        The top level container for all the plot elements.

    ax: :py:class:`matplotlib.axes.Axes`
        A single :py:class:`matplotlib.axes.Axes` object.

    Raises
    ------
    ValueError
        cm_max value should be an integer.
    ValueError
        cm_step value should be an integer.
    """

    if not isinstance(cm_max, int):
        raise ValueError("cm_max value should be an integer.")
    if not isinstance(cm_step, int):
        raise ValueError("cm_step value should be an integer.")

    def _fmt_ticks(x, int_max):
        """Function to format the integers at the side of the colormap"""
        if int(x) == int_max:
            return f">{int(x)}"
        else:
            return f"{int(x)}"

    # Color and marker size settings
    color_results_full = "#fe9901"
    # here only used for patch color in legend, should be "highest" color in cmap
    color_results_resampling = "#007972"
    markersize_full = 32
    # markersize_resampled = 16 # not necessary, heatmap

    # norm is a class which, when called, can normalize data into the
    # [0.0, 1.0] interval.
    vmin = 0
    vmax = 20
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    # Choose a colormap
    # cm = matplotlib.cm.Blues
    cm = _plot_utils._gridlib_cm()

    # Create a ScalarMappable and initialize a data structure
    sm = matplotlib.cm.ScalarMappable(cmap=cm, norm=norm)

    # Create the log values for the edges on the x-axis
    k_val = np.copy(k_full)

    if scale == "log":
        k_val = np.log10(k_val)

    # Middle between different k values on the logarithmic scale
    mid = (k_val[:-1] + k_val[1:]) / 2.0
    left_val = np.array([mid[0] - np.abs(mid[0] - mid[1])])
    right_val = np.array([mid[-1] + np.abs(mid[0] - mid[1])])
    k_val_edges = np.concatenate((left_val, mid, right_val), axis=0)

    # Create the edges for the x-axis
    if scale == "log":
        bin_edges_x = np.power(10.0, k_val_edges)
    else:
        bin_edges_x = np.copy(k_val_edges)

    bin_edges_y = np.linspace(0.0000001, 1, 101)

    # Create the figure
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    # Create the location for the colormap (cmap)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    # Plot the heat map
    ax.hist2d(
        k_resampled,
        weight_resampled,
        bins=[bin_edges_x, bin_edges_y],
        cmin=1,
        cmap=cm,
        vmin=vmin,
        vmax=vmax,
    )

    # Plot the full results
    # Only plot the weights that are above the threshold
    idx = weight_full >= threshold
    ax.scatter(
        k_full[idx],
        weight_full[idx],
        s=markersize_full,
        color=color_results_full,
        linewidth=1,
        facecolors="none",
    )

    if scale == "log":
        # Set the scale and x-Axis limits (adjust in log scale)
        ax.set_xscale("log")

    # Axis limits
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Create the side colorbar
    # 0 - 20 integers, steps of 2
    # cm_max = 20
    # cm_step = 2
    ticks = np.arange(0, cm_max + 0.1, step=cm_step)
    cbar = fig.colorbar(sm, cax=cax, ticks=ticks)
    cbar.ax.set_yticklabels([_fmt_ticks(i, cm_max) for i in ticks])

    if add_legend:
        # Legend
        legend_elements = [
            Patch(facecolor=color_results_resampling, label="resampled data"),
            Line2D(
                [0],
                [0],
                marker="o",
                linewidth=0,
                color=color_results_full,
                markerfacecolor="none",
                markersize=np.sqrt(markersize_full),
                label="full data",
            ),
        ]
        ax.legend(handles=legend_elements, loc="upper left")

    return fig, ax


def event_spectrum_heatmap(
    fit_result_full: Dict[str, Dict[str, Union[np.ndarray, float]]],
    fit_results_resampled: List[Dict[str, Dict[str, Union[np.ndarray, float]]]],
    fit_key: str = "grid",
    scale: str = "log",
    threshold: float = 10e-6,
    xlim: Tuple[float, float] = None,
    ylim: Tuple[float, float] = None,
    figsize: Tuple[float, float] = (6, 4),
    cm_max: int = 20,
    cm_step: int = 2,
    add_legend: bool = True,
):
    """
    Function plots the event spectrum heatmap for resampled data and plots the full
    data points as circles.

    Parameters
    ----------
    fit_result_full : Dict[str, Dict[str, Union[np.ndarray, float]]]
        A dictionary mapping keys (fitting procedure) to the corresponding fit results
        for the full data.
        For example::

            {
                "grid": {
                    "k": array([1.00000000e-03, 1.04737090e-03, ...]),
                    "s": array([3.85818587e-17, 6.42847878e-18, ...]),
                    "a": 0.010564217803906671,
                    "loss": 0.004705659331508584,
                },
            }

    fit_results_resampled : List[Dict[str, Dict[str, Union[np.ndarray, float]]]]
        A list consisting of dictionaries mapping keys (fitting procedure) to the
        corresponding fit results for the resampled data.
        For example::

            [
                {
                    "grid": {
                        "k": array([1.00000000e-03, 1.04737090e-03, ...]),
                        "s": array([3.85818587e-17, 6.42847878e-18, ...]),
                        "a": 0.010564217803906671,
                        "loss": 0.004705659331508584,
                    },
                },
                ...
            ]
    fit_key : str, optional
        The mapping key (fitting procedure) used to plot the resampling results from,
        by default "grid".
    scale : {"log", "linear"}, optional
        The scale of the x-axis. If scale is set to "log" than the x-axis will be
        logarithmic. If scale is set to "linear", the x-axis will be linear, by default
        "log".
    threshold : float, optional
        Minimum weight value that is shown for full data spectrum, by default 10e-6.
    xlim : Tuple[float, float], optional
        A tuple setting the x-axis limits. If the value is set to None, there are no
        limits, by default None.
    ylim : Tuple[float, float], optional
        A tuple setting the y-axis limits. If the value is set to None, there are no
        limits, by default None.
    figsize : Tuple[float, float], optional
        Width, height of the figure in inches, default is (6, 4).
    cm_max : int, optional
        Maximum value of the colormap/colorbar, by default 20.
    cm_step : int, optional
        Step size of the colormap, by default 2.
    add_legend : bool, optional
        If True, a legend is added to the figure, by default True.

    Returns
    -------
    fig: :py:class:`matplotlib.figure.Figure`
        The top level container for all the plot elements.

    ax: :py:class:`matplotlib.axes.Axes`
        A single :py:class:`matplotlib.axes.Axes` object.
    """

    # Full data results
    k_full = fit_result_full[fit_key]["k"]
    s_full = fit_result_full[fit_key]["s"]

    # Resampled data results
    # shape = (number of times data resampled, number of k values (dissociation rate value))
    k_resampled = np.zeros(
        (len(fit_results_resampled), fit_result_full[fit_key]["k"].shape[0]),
        dtype=np.float64,
    )
    s_resampled = np.zeros(
        (len(fit_results_resampled), fit_result_full[fit_key]["k"].shape[0]),
        dtype=np.float64,
    )

    for i in range(len(fit_results_resampled)):
        results = fit_results_resampled[i]
        k_resampled[i, :] = results[fit_key]["k"]
        s_resampled[i, :] = results[fit_key]["s"]

    # Create the base heatmap
    fig, ax = _base_heatmap(
        k_full,
        s_full,
        k_resampled.flatten(),
        s_resampled.flatten(),
        scale=scale,
        threshold=threshold,
        xlim=xlim,
        ylim=ylim,
        figsize=figsize,
        cm_max=cm_max,
        cm_step=cm_step,
        add_legend=add_legend,
    )

    # Add the appropriate labels
    ax.set_xlabel("dissociation rate (1/s)")
    ax.set_ylabel("event spectrum")

    return fig, ax


def state_spectrum_heatmap(
    fit_result_full: Dict[str, Union[np.ndarray, float]],
    fit_results_resampled: List[Dict[str, Union[np.ndarray, float]]],
    fit_key: str = "grid",
    scale: str = "log",
    threshold: float = 10e-6,
    xlim: Tuple[float, float] = None,
    ylim: Tuple[float, float] = None,
    figsize: Tuple[float, float] = (6, 4),
    cm_max: int = 20,
    cm_step: int = 2,
    add_legend: bool = True,
):
    """
    Function plots the state spectrum heatmap for resampled data and plots the full
    data points as circles.

    Parameters
    ----------
    fit_result_full : Dict[str, Dict[str, Union[np.ndarray, float]]]
        A dictionary mapping keys (fitting procedure) to the corresponding fit results
        for the full data.
        For example::

            {
                "grid": {
                    "k": array([1.00000000e-03, 1.04737090e-03, ...]),
                    "s": array([3.85818587e-17, 6.42847878e-18, ...]),
                    "a": 0.010564217803906671,
                    "loss": 0.004705659331508584,
                },
            }

    fit_results_resampled : List[Dict[str, Dict[str, Union[np.ndarray, float]]]]
        A list consisting of dictionaries mapping keys (fitting procedure) to the
        corresponding fit results for the resampled data.
        For example::

            [
                {
                    "grid": {
                        "k": array([1.00000000e-03, 1.04737090e-03, ...]),
                        "s": array([3.85818587e-17, 6.42847878e-18, ...]),
                        "a": 0.010564217803906671,
                        "loss": 0.004705659331508584,
                    },
                },
                ...
            ]
    fit_key : str, optional
        The mapping key (fitting procedure) used to plot the resampling results from,
        by default "grid".
    scale : {"log", "linear"}, optional
        The scale of the x-axis. If scale is set to "log" than the x-axis will be
        logarithmic. If scale is set to "linear", the x-axis will be linear, by default
        "log".
    threshold : float, optional
        Minimum weight value that is shown for full data spectrum, by default 10e-6.
    xlim : Tuple[float, float], optional
        A tuple setting the x-axis limits. If the value is set to None, there are no
        limits, by default None.
    ylim : Tuple[float, float], optional
        A tuple setting the y-axis limits. If the value is set to None, there are no
        limits, by default None.
    figsize : Tuple[float, float], optional
        Width, height of the figure in inches, default is (6, 4).
    cm_max : int, optional
        Maximum value of the colormap/colorbar, by default 20.
    cm_step : int, optional
        Step size of the colormap, by default 2.
    add_legend : bool, optional
        If True, a legend is added to the figure, by default True.

    Returns
    -------
    fig: :py:class:`matplotlib.figure.Figure`
        The top level container for all the plot elements.

    ax: :py:class:`matplotlib.axes.Axes`
        A single :py:class:`matplotlib.axes.Axes` object.
    """

    # Full data results
    k_full = fit_result_full[fit_key]["k"]
    s_full = fit_result_full[fit_key]["s"]
    state_full = (1 / k_full) * s_full
    state_full = state_full / np.sum(state_full)  # normalization

    # Resampled data results
    # shape = (number of times data resampled, number of k values (dissociation rate value))
    k_resampled = np.zeros(
        (len(fit_results_resampled), fit_result_full[fit_key]["k"].shape[0]),
        dtype=np.float64,
    )
    s_resampled = np.zeros(
        (len(fit_results_resampled), fit_result_full[fit_key]["k"].shape[0]),
        dtype=np.float64,
    )

    for i in range(len(fit_results_resampled)):
        results = fit_results_resampled[i]
        k_resampled[i, :] = results[fit_key]["k"]
        s_resampled[i, :] = results[fit_key]["s"]

    # Create the state array
    state_resampled = (1 / k_resampled) * s_resampled
    # Normalization
    state_resampled = state_resampled / np.sum(state_resampled, axis=1)[:, None]

    # Create the base heatmap
    fig, ax = _base_heatmap(
        k_full,
        state_full,
        k_resampled.flatten(),
        state_resampled.flatten(),
        scale=scale,
        threshold=threshold,
        xlim=xlim,
        ylim=ylim,
        figsize=figsize,
        cm_max=cm_max,
        cm_step=cm_step,
        add_legend=add_legend,
    )

    # Labels
    ax.set_xlabel("dissociation rate (1/s)")
    ax.set_ylabel("state spectrum")

    return fig, ax
