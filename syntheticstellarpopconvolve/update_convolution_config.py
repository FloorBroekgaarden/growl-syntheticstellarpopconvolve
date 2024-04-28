"""
Function to add some information to the config
"""

import numpy as np


def update_convolution_config(config):
    """
    Function to calculate some extra quantities based on input
    """

    #
    config["logger"].debug("Updating configuration")

    # Calculate convolution-time quantities
    config["convolution_time_bin_centers"] = (
        config["convolution_time_bin_edges"][1:]
        + config["convolution_time_bin_edges"][:-1]
    ) / 2
    config["convolution_time_bin_sizes"] = np.diff(config["convolution_time_bin_edges"])

    return config
