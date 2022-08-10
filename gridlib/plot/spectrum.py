"""
Module with functions to plot event spectrum and state spectrum.
"""
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredText


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
        k[idx], weight[idx], linestyle="None", marker="o", markersize=3, color=color
    )
    ax.vlines(
        k[idx], np.zeros(k.shape[0])[idx], weight[idx], linewidth=0.75, color=color
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
    fit_results_grid,
    scale: str = "log",
    threshold: float = 0.0,
    xlim: Tuple[float, float] = None,
    ylim: Tuple[float, float] = None,
    figsize: Tuple[float, float] = (10, 6),
    color="#fe9901",
):

    k = fit_results_grid["k"]
    weight = fit_results_grid["s"]
    bleaching_number = fit_results_grid["a"]

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
    fit_results_grid,
    scale: str = "log",
    threshold: float = 0.0,
    xlim: Tuple[float, float] = None,
    ylim: Tuple[float, float] = None,
    figsize: Tuple[float, float] = (10, 6),
    color="#fe9901",
):

    k = fit_results_grid["k"]
    weight = fit_results_grid["s"]
    bleaching_number = fit_results_grid["a"]

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
