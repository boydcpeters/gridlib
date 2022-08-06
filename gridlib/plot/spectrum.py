"""
Module with functions to plot event spectrum and state spectrum.
"""
import pathlib
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredText

# TODO: use kwargs structure here as well


def _base_spectrum(
    k: np.ndarray,
    weight: np.ndarray,
    figsize: Tuple[float, float] = (10, 8),
    xlim: Tuple[float, float] = None,
    ylim: Tuple[float, float] = None,
):

    # Color and marker size settings
    color_results = "#fe9901"
    markersize = 16

    fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    # Plot the results
    idx = weight >= 0.00001
    ax1.scatter(k[idx], weight[idx], s=markersize, color=color_results)
    ax1.vlines(
        k[idx], np.zeros(k.shape[0])[idx], weight[idx], linewidth=1, color=color_results
    )

    ax1.set_xscale("log")

    # Axis limits
    if xlim is not None:
        ax1.set_xlim(xlim)
    if ylim is not None:
        ax1.set_ylim(ylim)

    return fig1, ax1


def event_spectrum(
    fit_results_grid,
    figsize: Tuple[float, float] = (10, 8),
    xlim: Tuple[float, float] = None,
    ylim: Tuple[float, float] = None,
    path_save: Union[str, pathlib.Path] = None,
):

    # Create a Path() from path_save if not None and it is a str
    if path_save is not None and isinstance(path_save, str):
        path_save = pathlib.Path(path_save)

    k = fit_results_grid["k"]
    weight = fit_results_grid["s"]
    bleaching_number = fit_results_grid["a"]

    fig1, ax1 = _base_spectrum(k, weight, figsize=figsize, xlim=xlim, ylim=ylim)

    # Labels
    ax1.set_xlabel("dissociation rate (1/s)")
    ax1.set_ylabel("event spectrum")

    # add the bleaching number in the plot
    # https://stackoverflow.com/questions/23112903/matplotlib-text-boxes-automatic-position
    anchored_text = AnchoredText(
        f"a = {bleaching_number:.5f}", loc="center left", frameon=False
    )
    ax1.add_artist(anchored_text)

    if path_save is not None:
        fig1.savefig(path_save, bbox_inches="tight", dpi=200)
        plt.close(fig1)


def state_spectrum(
    fit_results_grid,
    figsize: Tuple[float, float] = (10, 8),
    xlim: Tuple[float, float] = None,
    ylim: Tuple[float, float] = None,
    path_save: Union[str, pathlib.Path] = None,
):

    # Create a Path() from path_save if not None and it is a str
    if path_save is not None and isinstance(path_save, str):
        path_save = pathlib.Path(path_save)

    k = fit_results_grid["k"]
    weight = fit_results_grid["s"]
    bleaching_number = fit_results_grid["a"]

    state = (1 / k) * weight
    state = state / np.sum(state)  # normalization

    fig1, ax1 = _base_spectrum(k, state, figsize=figsize, xlim=xlim, ylim=ylim)

    # Labels
    ax1.set_xlabel("dissociation rate (1/s)")
    ax1.set_ylabel("state spectrum")

    # add the bleaching number in the plot
    # https://stackoverflow.com/questions/23112903/matplotlib-text-boxes-automatic-position
    anchored_text = AnchoredText(
        f"a = {bleaching_number:.5f}", loc="center right", frameon=False
    )
    ax1.add_artist(anchored_text)

    if path_save is not None:
        fig1.savefig(path_save, bbox_inches="tight", dpi=200)
        plt.close(fig1)
