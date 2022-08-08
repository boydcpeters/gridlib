"""
Module with functions to plot event spectrum and state spectrum.
"""
import pathlib
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredText


def _base_spectrum(
    k: np.ndarray,
    weight: np.ndarray,
    scale: str = "log",
    threshold: float = 0.001,
    figsize: Tuple[float, float] = (10, 6),
    xlim: Tuple[float, float] = None,
    ylim: Tuple[float, float] = None,
    color="#fe9901",
):

    fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    # Plot the results
    idx = weight >= threshold
    ax1.plot(
        k[idx], weight[idx], linestyle="None", marker="o", markersize=3, color=color
    )
    ax1.vlines(
        k[idx], np.zeros(k.shape[0])[idx], weight[idx], linewidth=0.75, color=color
    )

    if scale == "log":
        ax1.set_xscale("log")

    # Axis limits
    if xlim is not None:
        ax1.set_xlim(xlim)
    if ylim is not None:
        ax1.set_ylim(ylim)

    return fig1, ax1


def event_spectrum(
    fit_results_grid,
    scale: str = "log",
    threshold: float = 0.001,
    figsize: Tuple[float, float] = (10, 6),
    xlim: Tuple[float, float] = None,
    ylim: Tuple[float, float] = None,
    path_save: Union[str, pathlib.Path] = None,
    color="#fe9901",
):

    # Create a Path() from path_save if not None and it is a str
    if path_save is not None and isinstance(path_save, str):
        path_save = pathlib.Path(path_save)

    k = fit_results_grid["k"]
    weight = fit_results_grid["s"]
    bleaching_number = fit_results_grid["a"]

    fig1, ax1 = _base_spectrum(
        k,
        weight,
        scale=scale,
        threshold=threshold,
        figsize=figsize,
        xlim=xlim,
        ylim=ylim,
        color=color,
    )

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
    scale: str = "log",
    threshold: float = 0.001,
    figsize: Tuple[float, float] = (10, 6),
    xlim: Tuple[float, float] = None,
    ylim: Tuple[float, float] = None,
    path_save: Union[str, pathlib.Path] = None,
    color="#fe9901",
):

    # Create a Path() from path_save if not None and it is a str
    if path_save is not None and isinstance(path_save, str):
        path_save = pathlib.Path(path_save)

    k = fit_results_grid["k"]
    weight = fit_results_grid["s"]
    bleaching_number = fit_results_grid["a"]

    state = (1 / k) * weight
    state = state / np.sum(state)  # normalization

    fig1, ax1 = _base_spectrum(
        k,
        state,
        scale=scale,
        threshold=threshold,
        figsize=figsize,
        xlim=xlim,
        ylim=ylim,
        color=color,
    )

    # Labels
    ax1.set_xlabel("dissociation rate (1/s)")
    ax1.set_ylabel("state spectrum")

    # add the bleaching number in the plot
    # https://stackoverflow.com/questions/23112903/matplotlib-text-boxes-automatic-position
    anchored_text = AnchoredText(
        f"a = {bleaching_number:.5f}",
        loc="center right",
        frameon=False,
    )
    ax1.add_artist(anchored_text)

    if path_save is not None:
        fig1.savefig(path_save, bbox_inches="tight", dpi=200)
        plt.close(fig1)
