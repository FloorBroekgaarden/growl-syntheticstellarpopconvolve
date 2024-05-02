"""
Function to extract the population settings from the hdf5 file and load it into the config
"""

import json

import h5py


def extract_population_settings(config):
    """
    Function to extract population settings and load them into config
    """

    #
    config["logger"].debug(
        "Extracting population-settings from hdf5 file and loading into config"
    )

    with h5py.File(config["output_filename"], "r") as output_hdf5file:
        population_settings = json.loads(output_hdf5file["config/population"][()])

        if "binary_c_help_all" in population_settings.keys():
            del population_settings["binary_c_help_all"]

        config["population_settings"] = population_settings

    return config
