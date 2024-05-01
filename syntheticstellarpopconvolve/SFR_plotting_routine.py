"""
Functionality to plot a starformation distribution
"""

# from syntheticstellarpopconvolve.convolution_general_functions import (
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
