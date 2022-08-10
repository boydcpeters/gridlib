"""
Module with functions to plot event spectrum and state spectrum.
"""
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredText

from ..plot import _plot_utils


def _base_spectrum(
    k: np.ndarray,
    weight: np.ndarray,
    scale: str = "log",
    threshold: float = 0.0,
    xlim: Tuple[float, float] = None,
    ylim: Tuple[float, float] = None,
    figsize: Tuple[float, float] = (10, 6),
    color="#fe9901",
):

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    # Plot the results
    idx = weight >= threshold
    ax.plot(
        k[idx],
        weight[idx],
        linestyle="None",
        marker="o",
        markersize=3,
        color=color,
    )
    ax.vlines(
        k[idx],
        np.zeros(k.shape[0])[idx],
        weight[idx],
        linewidth=0.75,
        color=color,
    )

    if scale == "log":
        ax.set_xscale("log")

    # Axis limits
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    return fig, ax


def event_spectrum(
    fit_results,
    scale: str = "log",
    threshold: float = 0.0,
    xlim: Tuple[float, float] = None,
    ylim: Tuple[float, float] = None,
    figsize: Tuple[float, float] = (10, 6),
    color="#fe9901",
):

    k = fit_results["k"]
    weight = fit_results["s"]
    bleaching_number = fit_results["a"]

    fig, ax = _base_spectrum(
        k,
        weight,
        scale=scale,
        threshold=threshold,
        xlim=xlim,
        ylim=ylim,
        figsize=figsize,
        color=color,
    )

    # Labels
    ax.set_xlabel("dissociation rate (1/s)")
    ax.set_ylabel("event spectrum")

    # add the bleaching number in the plot
    # https://stackoverflow.com/questions/23112903/matplotlib-text-boxes-automatic-position
    anchored_text = AnchoredText(
        f"a = {bleaching_number:.5f}", loc="center left", frameon=False
    )
    ax.add_artist(anchored_text)

    return fig, ax


def state_spectrum(
    fit_results,
    scale: str = "log",
    threshold: float = 0.0,
    xlim: Tuple[float, float] = None,
    ylim: Tuple[float, float] = None,
    figsize: Tuple[float, float] = (10, 6),
    color="#fe9901",
):

    k = fit_results["k"]
    weight = fit_results["s"]
    bleaching_number = fit_results["a"]

    state = (1 / k) * weight
    state = state / np.sum(state)  # normalization

    fig, ax = _base_spectrum(
        k,
        state,
        scale=scale,
        threshold=threshold,
        xlim=xlim,
        ylim=ylim,
        figsize=figsize,
        color=color,
    )

    # Labels
    ax.set_xlabel("dissociation rate (1/s)")
    ax.set_ylabel("state spectrum")

    # add the bleaching number in the plot
    # https://stackoverflow.com/questions/23112903/matplotlib-text-boxes-automatic-position
    anchored_text = AnchoredText(
        f"a = {bleaching_number:.5f}",
        loc="center right",
        frameon=False,
    )
    ax.add_artist(anchored_text)

    return fig, ax


def _base_spectrum_with_multi_exp(
    key_to_k_and_weight,
    scale: str = "log",
    threshold: float = 0.0,
    xlim: Tuple[float, float] = None,
    ylim: Tuple[float, float] = None,
    figsize: Tuple[float, float] = (10, 6),
    color=None,
):
    """"""

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
            linewidth=0.75,
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

    # Legend
    # Remove duplicate labels
    # Implementation from: https://stackoverflow.com/questions/13588920/stop-matplotlib-repeating-labels-in-legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())  # , loc="lower left")

    return fig, ax


# Important: the weight_single_exp is overwritten from 1 to 0.2 for visual reasons, see comment in code.


def event_spectrum_with_multi_exp(
    fit_values,
    scale: str = "log",
    threshold: float = 0.0,
    xlim: Tuple[float, float] = None,
    ylim: Tuple[float, float] = None,
    figsize: Tuple[float, float] = (10, 6),
    color=None,
):

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

    fig, ax = _base_spectrum_with_multi_exp(
        key_to_k_and_weight,
        scale=scale,
        threshold=threshold,
        xlim=xlim,
        ylim=ylim,
        figsize=figsize,
        color=color,
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
