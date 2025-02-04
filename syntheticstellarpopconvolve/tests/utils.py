"""
Util functions for the tests
"""

import copy
import json
import os

import astropy.units as u
import h5py
import numpy as np
import pkg_resources

from syntheticstellarpopconvolve import default_convolution_config


class Boilerplate:

    def __init__(self):
        """ """

    def setup(self, name, tmp_dir, add_population_settings, sfr_unit):
        #
        input_hdf5_filename = os.path.join(tmp_dir, "input_hdf5_{}.h5".format(name))
        output_hdf5_filename = os.path.join(tmp_dir, "output_hdf5_{}.h5".format(name))

        #############
        # create input HDF5 file
        with h5py.File(input_hdf5_filename, "w") as input_hdf5_file:

            ######################
            # Create groups
            input_hdf5_file.create_group("input_data")
            input_hdf5_file.create_group("config")

            ###############
            # Readout population settings
            if add_population_settings:
                population_settings_filename = pkg_resources.resource_filename(
                    "syntheticstellarpopconvolve",
                    "example_data/example_population_settings.json",
                )

                with open(population_settings_filename, "r") as f:
                    population_settings = json.loads(f.read())

                # Delete some stuff from the settings
                del population_settings["population_settings"]["bse_options"][
                    "metallicity"
                ]

                # Write population config to file
                input_hdf5_file.create_dataset(
                    "config/population", data=json.dumps(population_settings)
                )
            else:
                input_hdf5_file.create_dataset("config/population", data=json.dumps({}))

        #
        self.convolution_config = copy.copy(default_convolution_config)

        # Set up SFR
        self.convolution_config["SFR_info"] = {
            "lookback_time_bin_edges": np.array([0, 1, 2, 3, 4, 5]) * u.Gyr,
            "starformation_rate_array": np.array([2, 1, 1, 1, 1]) * sfr_unit,
        }

        # set up convolution bins
        self.convolution_config["convolution_lookback_time_bin_edges"] = (
            np.array([0, 1, 2, 3, 4]) * u.Gyr
        )

        # lookback time convolution only
        self.convolution_config["time_type"] = "lookback_time"

        #
        self.convolution_config["input_filename"] = input_hdf5_filename
        self.convolution_config["output_filename"] = output_hdf5_filename

        self.convolution_config["redshift_interpolator_data_output_filename"] = (
            os.path.join(tmp_dir, "interpolator_dict.p")
        )

        #
        self.convolution_config["tmp_dir"] = os.path.join(
            tmp_dir, "tmp_{}".format(name)
        )
