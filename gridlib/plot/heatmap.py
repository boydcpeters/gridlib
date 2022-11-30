"""
Module with functions to plot event heatmap and state heatmap.
"""
from typing import Tuple, Union, List, Dict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable

from . import _plot_utils


def _base_heatmap(
    k_full: np.ndarray,
    weight_full: np.ndarray,
    k_resampled: np.ndarray,
    weight_resampled: np.ndarray,
    scale: str = "log",
    threshold: float = 0.0001,
    xlim: Tuple[float, float] = None,
    ylim: Tuple[float, float] = None,
    figsize: Tuple[float, float] = (8, 5),
    add_legend: bool = True,
):
    """weights can be either event weights or state weights, arrays should be flattened"""

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
    int_max_cm = 20
    ticks = np.arange(0, int_max_cm + 0.1, step=2)
    cbar = fig.colorbar(sm, cax=cax, ticks=ticks)
    cbar.ax.set_yticklabels([_fmt_ticks(i, int_max_cm) for i in ticks])

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
    fit_result_full: Dict[str, Union[np.array, float]],
    fit_results_resampled: List[Dict[str, Union[np.array, float]]],
    scale: str = "log",
    threshold: float = 0.0001,
    xlim: Tuple[float, float] = None,
    ylim: Tuple[float, float] = None,
    figsize: Tuple[float, float] = (8, 5),
    add_legend: bool = True,
):

    # Full data results
    k_full = fit_result_full["k"]
    s_full = fit_result_full["s"]

    # Resampled data results
    # shape = (number of times data resampled, number of k values (dissociation rate value))
    k_resampled = np.zeros(
        (len(fit_results_resampled), fit_result_full["k"].shape[0]), dtype=np.float64
    )
    s_resampled = np.zeros(
        (len(fit_results_resampled), fit_result_full["k"].shape[0]), dtype=np.float64
    )

    for i in range(len(fit_results_resampled)):
        results = fit_results_resampled[i]
        k_resampled[i, :] = results["k"]
        s_resampled[i, :] = results["s"]

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
        add_legend=add_legend,
    )

    # Add the appropriate labels
    ax.set_xlabel("dissociation rate (1/s)")
    ax.set_ylabel("event spectrum")

    return fig, ax


def state_spectrum_heatmap(
    fit_result_full: Dict[str, Union[np.array, float]],
    fit_results_resampled: List[Dict[str, Union[np.array, float]]],
    scale: str = "log",
    threshold: float = 0.0001,
    xlim: Tuple[float, float] = None,
    ylim: Tuple[float, float] = None,
    figsize: Tuple[float, float] = (8, 5),
    add_legend: bool = True,
):

    # Full data results
    k_full = fit_result_full["k"]
    s_full = fit_result_full["s"]
    state_full = (1 / k_full) * s_full
    state_full = state_full / np.sum(state_full)  # normalization

    # Resampled data results
    # shape = (number of times data resampled, number of k values (dissociation rate value))
    k_resampled = np.zeros(
        (len(fit_results_resampled), fit_result_full["k"].shape[0]), dtype=np.float64
    )
    s_resampled = np.zeros(
        (len(fit_results_resampled), fit_result_full["k"].shape[0]), dtype=np.float64
    )

    for i in range(len(fit_results_resampled)):
        results = fit_results_resampled[i]
        k_resampled[i, :] = results["k"]
        s_resampled[i, :] = results["s"]

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
        add_legend=add_legend,
    )

    # Labels
    ax.set_xlabel("dissociation rate (1/s)")
    ax.set_ylabel("state spectrum")

    return fig, ax
