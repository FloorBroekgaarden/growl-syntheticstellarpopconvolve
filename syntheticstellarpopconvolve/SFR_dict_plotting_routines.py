"""
Functionality to plot the content of a SFR dict

TODO: handle units properly
"""

import astropy.units as u
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from syntheticstellarpopconvolve.general_functions import calculate_bincenters


def plot_SFR_dict(config, SFR_dict, fig=None, return_fig=False):
    """ """

    ####
    #
    if fig is None:
        fig = plt.figure(figsize=(20, 20))

    axes_dict = {}

    #
    gs = fig.add_gridspec(nrows=2, ncols=1)

    if "metallicity_distribution_array" in SFR_dict:

        # create axes
        axes_dict["ax_starformation_rate"] = fig.add_subplot(gs[:1, :])
        axes_dict["ax_metallicity_distribution"] = fig.add_subplot(gs[1:, :])
    else:
        axes_dict["ax_starformation_rate"] = fig.add_subplot(gs[:, :])

    ############
    # Plot sfr array
    if config["time_type"] == "lookback_time":
        time_array = calculate_bincenters(sfr_dict["lookback_time_bin_edges"])
    elif config["time_type"] == "redshift":
        time_array = calculate_bincenters(sfr_dict["redshift_bin_edges"])
    else:
        raise ValueError(
            "either 'lookback_time' or 'redshift' has to be present in the sfr_dict"
        )

    #
    axes_dict["ax_starformation_rate"].plot(
        time_array, sfr_dict["starformation_rate_array"]
    )

    #
    if return_fig:
        return fig, axes_dict
    plt.show()


# from syntheticstellarpopconvolve.general_functions import (
#     calculate_bincenters,
# )

# def plot_sfr(sfr_dict, fig=None, return_fig=False):  # DH0001
#     """
#     Function to plot a starformation distribution
#     """

#     # TODO: if only the SFR array is present, plot that

#     ############
#     # Plot sfr array
#     if "lookback_time" in sfr_dict:
#         time_array = calculate_bincenters(sfr_dict["lookback_time_bin_edges"])
#     elif "redshift" in sfr_dict:
#         time_array = calculate_bincenters(sfr_dict["redshift_bin_edges"])
#     else:
#         raise ValueError(
#             "either 'lookback_time' or 'redshift' has to be present in the sfr_dict"
#         )

#     ax_sfr.plot(time_array, sfr_dict["starformation_array"])

#     ###########
#     # Plot mssfr grid


if __name__ == "__main__":

    # construct the sfr-dict (NOTE: this uses absolute SFR, not metallicity dependent)
    sfr_dict = {}
    sfr_dict["lookback_time_bin_edges"] = (np.arange(3, 6, 1) * u.Gyr).to(u.yr)
    sfr_dict["starformation_rate_array"] = (
        0.25 * np.ones(sfr_dict["lookback_time_bin_edges"].shape[0] - 1) * u.Msun / u.yr
    )  # example of a constant star-formation rate. this could be anything of course.

    plot_SFR_dict(config={"time_type": "lookback_time"}, SFR_dict=sfr_dict)
