import pathlib
import re
from typing import Dict, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredText

import gridlib.data_utils as data_utils


def fmt_t_tl(t_tl):
    # pattern to match time pattern in the files, eg. 100ms or 2s
    time_pattern = re.compile(
        r"[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)", flags=re.IGNORECASE
    )
    unit_pattern = re.compile("(ms|s)", flags=re.IGNORECASE)

    # Match time pattern
    m_time = time_pattern.search(t_tl)

    # Match unit pattern
    m_unit = unit_pattern.search(t_tl)

    # get the number value of the match, eg. 50, 100, 390, 4
    t_tl_val = float(m_time.group(0))
    # get the unit value of the match, eg. ms or s
    t_tl_unit = m_unit.group(0)

    # Calculate the frame cycle time
    if t_tl_unit == "ms":
        frame_cycle_time = t_tl_val / 1000.0
    elif t_tl_unit == "s":
        frame_cycle_time = t_tl_val

    if frame_cycle_time < 1:
        return rf"${frame_cycle_time:.2f}\,\mathrm{{s}}$"
    elif frame_cycle_time % 1 == 0:
        return rf"${frame_cycle_time:.0f}\,\mathrm{{s}}$"
    else:
        return rf"${frame_cycle_time:.2f}\,\mathrm{{s}}$"


def create_base_spectrum(
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


def plot_event_spectrum(
    fit_values_grid,
    figsize: Tuple[float, float] = (10, 8),
    xlim: Tuple[float, float] = None,
    ylim: Tuple[float, float] = None,
    path_save: Union[str, pathlib.Path] = None,
):

    # Create a Path() from path_save if not None and it is a str
    if path_save is not None and isinstance(path_save, str):
        path_save = pathlib.Path(path_save)

    k = fit_values_grid["k"]
    weight = fit_values_grid["S"]
    bleaching_number = fit_values_grid["a1"]

    fig1, ax1 = create_base_spectrum(k, weight, figsize=figsize, xlim=xlim, ylim=ylim)

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


def plot_state_spectrum(
    fit_values_grid,
    figsize: Tuple[float, float] = (10, 8),
    xlim: Tuple[float, float] = None,
    ylim: Tuple[float, float] = None,
    path_save: Union[str, pathlib.Path] = None,
):

    # Create a Path() from path_save if not None and it is a str
    if path_save is not None and isinstance(path_save, str):
        path_save = pathlib.Path(path_save)

    k = fit_values_grid["k"]
    weight = fit_values_grid["S"]
    bleaching_number = fit_values_grid["a1"]

    state = (1 / k) * weight
    state = state / np.sum(state)  # normalization

    fig1, ax1 = create_base_spectrum(k, state, figsize=figsize, xlim=xlim, ylim=ylim)

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


def create_base_sf_data_vs_multi_exp(
    data_sf: Dict[str, Dict[str, np.ndarray]],
    data_sf_multi_exp: Dict[str, Dict[str, np.ndarray]],
    figsize: Tuple[float, float] = (10, 8),
):
    """Function plots the survival function of the true data and the multi-exponential curves.

    Parameters
    ----------
    data_sf: Dict[str, Dict[str, np.ndarray]]
        Data of the survival function from real data with the following data structure:
        {
            f"{t_tl}": {
                "time": np.ndarray with all the time values,
                "value": np.ndarray with all the survival function values corresponding to the
                    respective time value
            }
        }
    data_sf_multi_exp: Dict[str, Dict[str, np.ndarray]]
        Data of the computed GRID survival functions. The dictionary structure is as follows:
        {
            f"{t_tl}": {
                "time": np.ndarray with all the time values,
                "value": np.ndarray with all the survival function values corresponding to the
                    respective time value
            }
        }
    """
    # Color and linewidth size settings
    color_data = "#007972"
    color_multi_exp = "#fe9901"
    linewidth_data = 1
    linewidth_multi_exp = 1

    # Process data_sf
    data_sf = data_utils.process_data(data_sf)

    fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    # Set to store all the text positions in the graph
    text_points = set()

    # Sort the time-lapse conditions from low to high, required for proper
    # text positions
    for t_tl in sorted(data_sf.keys()):
        time = data_sf[t_tl]["time"]
        value = data_sf[t_tl]["value"]

        # Survival probability values
        value_prob = value / value[0]

        ax1.loglog(time, value_prob, color=color_data, linewidth=linewidth_data)

        time_multi_exp = data_sf_multi_exp[t_tl]["time"]
        value_multi_exp = data_sf_multi_exp[t_tl]["value"]
        ax1.loglog(
            time_multi_exp,
            value_multi_exp,
            color=color_multi_exp,
            linestyle="dashed",
            linewidth=linewidth_multi_exp,
        )

        # Create the new point position for the text
        text_point = (time[0] * 0.6, value_prob[0] * 1.1)

        # Check if the new text will not overlap
        if text_point in text_points:
            text_point = (time[0] * 1.3, value_prob[0] * 1.1)

        # This is a hack-job and should not be a long term fix
        # TODO: fix this do it automatically
        a = fmt_t_tl(t_tl)
        if a == "$0.39\,\mathrm{s}$":
            text_point = (time[0] * 1.1, value_prob[0] * 1.1)

        ax1.text(
            text_point[0], text_point[1], fmt_t_tl(t_tl), color=color_data, fontsize=8
        )
        text_points.add(text_point)

    return fig1, ax1, color_data, color_multi_exp, linewidth_data, linewidth_multi_exp


def plot_sf_data_vs_grid(
    data_sf: Dict[str, Dict[str, np.ndarray]],
    data_sf_grid: Dict[str, Dict[str, np.ndarray]],
    figsize: Tuple[float, float] = (10, 8),
    xlim: Tuple[float, float] = None,
    ylim: Tuple[float, float] = None,
    path_save: Union[str, pathlib.Path] = None,
):
    """Function plots the survival function of the true data and the GRID curves.

    Parameters
    ----------
    data_sf: Dict[str, Dict[str, np.ndarray]]
        Data of the survival function from real data with the following data structure:
        {
            f"{t_tl}": {
                "time": np.ndarray with all the time values,
                "value": np.ndarray with all the survival function values corresponding to the
                    respective time value
            }
        }
    data_sf_grid: Dict[str, Dict[str, np.ndarray]]
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

    print(data_sf.keys())

    # Create a Path() from path_save if not None and it is a str
    if path_save is not None and isinstance(path_save, str):
        path_save = pathlib.Path(path_save)

    (
        fig1,
        ax1,
        color_data,
        color_grid,
        linewidth_data,
        linewidth_grid,
    ) = create_base_sf_data_vs_multi_exp(data_sf, data_sf_grid, figsize=figsize)

    # Legend
    legend_elements = [
        Line2D(
            [0],
            [0],
            color=color_grid,
            linestyle="dashed",
            linewidth=linewidth_grid,
            label="fit GRID",
        ),
        Line2D([0], [0], color=color_data, linewidth=linewidth_data, label="data"),
    ]
    ax1.legend(handles=legend_elements, loc="lower left")

    # Axis limits
    if xlim is not None:
        ax1.set_xlim(xlim)
    if ylim is not None:
        ax1.set_ylim(ylim)
    # ax1.set_xlim((10**-1, 10**3))
    # ax1.set_ylim((10**-5, 10**0))

    # Labels
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("survival function")

    if path_save is not None:
        fig1.savefig(path_save, bbox_inches="tight", dpi=200)
        plt.close(fig1)
