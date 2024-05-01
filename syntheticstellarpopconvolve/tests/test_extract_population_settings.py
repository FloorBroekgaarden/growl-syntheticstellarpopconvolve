"""
Testcases for extract_population_settings file
"""

import copy
import json
import os
import unittest

import astropy.units as u
import h5py
import numpy as np
import pandas as pd
import pkg_resources

from syntheticstellarpopconvolve import convolution_options_defaults
from syntheticstellarpopconvolve.convolution_general_functions import temp_dir
from syntheticstellarpopconvolve.extract_population_settings import (
    extract_population_settings,
)
from syntheticstellarpopconvolve.prepare_output_file import prepare_output_file

TMP_DIR = temp_dir(
    "tests", "tests_convolution", "tests_extract_population_settings", clean_path=True
)


class test_extract_population_settings(unittest.TestCase):
    """ """

    def setUp(self):
        #
        input_hdf5_filename = os.path.join(TMP_DIR, "input_hdf5_sfr_only.h5")
        output_hdf5_filename = os.path.join(TMP_DIR, "output_hdf5_sfr_only.h5")

        ##############
        # SET UP DATA
        self.dummy_data = {
            "delay_time": np.array([0, 1, 2, 3]),
            "probability": np.array([1, 2, 3, 4]),
        }
        dummy_df = pd.DataFrame.from_records(self.dummy_data)

        #############
        # create input HDF5 file
        with h5py.File(input_hdf5_filename, "w") as input_hdf5_file:

            ######################
            # Create groups
            input_hdf5_file.create_group("input_data")
            input_hdf5_file.create_group("input_data/events")
            input_hdf5_file.create_group("config")

            ###############
            # Readout population settings
            population_settings_filename = pkg_resources.resource_filename(
                "syntheticstellarpopconvolve",
                "example_data/example_population_settings.json",
            )

            with open(population_settings_filename, "r") as f:
                population_settings = json.loads(f.read())

            # Delete some stuff from the settings
            del population_settings["population_settings"]["bse_options"]["metallicity"]

            # Write population config to file
            input_hdf5_file.create_dataset(
                "config/population", data=json.dumps(population_settings)
            )

        ##############
        # Store data in pandas
        dummy_df.to_hdf(input_hdf5_filename, key="input_data/events/{}".format("dummy"))

        #
        self.convolution_config = copy.copy(convolution_options_defaults)

        # Set up SFR
        self.convolution_config["SFR_info"] = {
            "lookback_time_bin_edges": np.array([0, 1, 2, 3, 4, 5]),
            "starformation_array": np.array([1, 1, 1, 1, 1]) * u.Msun / u.yr / u.Gpc**3,
        }

        # set up convolution bins
        self.convolution_config["convolution_time_bin_edges"] = np.array(
            [0, 1, 2, 3, 4]
        )

        # lookback time convolution only
        self.convolution_config["time_type"] = "lookback_time"

        #
        self.convolution_config["input_filename"] = input_hdf5_filename
        self.convolution_config["output_filename"] = output_hdf5_filename

        self.convolution_config["redshift_interpolator_data_output_filename"] = (
            os.path.join(TMP_DIR, "interpolator_dict.p")
        )

        #
        self.convolution_config["convolution_instructions"] = [
            {
                "input_data_type": "event",
                "input_data_name": "dummy",
                "output_data_name": "dummy",
                "data_column_dict": {
                    "delay_time": "delay_time",
                    "yield_rate": "probability",
                },
                "ignore_metallicity": True,
            },
        ]

        #
        self.convolution_config["tmp_dir"] = os.path.join(TMP_DIR, "tmp")

        #
        prepare_output_file(config=self.convolution_config)

    def test(self):
        self.assertTrue("population_settings" not in self.convolution_config)
        self.convolution_config = extract_population_settings(self.convolution_config)
        self.assertTrue("population_settings" in self.convolution_config)
        self.assertTrue(
            "binary_c_help_all" not in self.convolution_config["population_settings"]
        )


if __name__ == "__main__":
    unittest.main()
