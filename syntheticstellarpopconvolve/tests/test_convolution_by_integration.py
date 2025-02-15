"""
Testcases for convolution_by_integration file
"""

import copy
import json
import logging
import os
import unittest

import astropy.units as u
import h5py
import numpy as np
import pandas as pd
import pkg_resources

from syntheticstellarpopconvolve import (
    default_convolution_config,
    default_convolution_instruction,
)
from syntheticstellarpopconvolve.check_and_update_convolution_config import (
    check_and_update_convolution_config,
)
from syntheticstellarpopconvolve.convolution_by_integration import (
    convolution_by_integration,
)
from syntheticstellarpopconvolve.convolve_populations import extract_data
from syntheticstellarpopconvolve.general_functions import temp_dir
from syntheticstellarpopconvolve.prepare_output_file import prepare_output_file

TMP_DIR = temp_dir(
    "tests", "tests_convolution", "tests_convolution_by_integration", clean_path=True
)


class test_convolution_by_integration(unittest.TestCase):
    """
    TODO: make a more complicated post convolution hook function test
    """

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
        dummy_df.to_hdf(input_hdf5_filename, key="input_data/{}".format("dummy"))

        #
        self.convolution_config = copy.copy(default_convolution_config)
        self.convolution_config["logger"].setLevel(logging.CRITICAL)

        # Set up SFR
        self.convolution_config["SFR_info"] = {
            "lookback_time_bin_edges": np.array([0, 1, 2, 3, 4, 5]) * u.yr,
            "starformation_rate_array": np.array([1, 1, 1, 1, 1])
            * u.Msun
            / u.yr
            / u.Gpc**3,
        }

        # set up convolution bins
        self.convolution_config["convolution_lookback_time_bin_edges"] = (
            np.array([0, 1, 2, 3, 4]) * u.yr
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
                **default_convolution_instruction,
                "input_data_name": "dummy",
                "output_data_name": "dummy",
                "convolution_type": "integrate",
                "data_column_dict": {
                    "delay_time": "delay_time",
                    "normalized_yield": "probability",
                },
                "ignore_metallicity": True,
            },
        ]

        #
        self.convolution_config["tmp_dir"] = os.path.join(TMP_DIR, "tmp")

        #
        check_and_update_convolution_config(self.convolution_config)

        #
        prepare_output_file(config=self.convolution_config)

    def test_convolution_by_integration_normal(self):
        #
        normal_convolution_instructions = {
            **default_convolution_instruction,
            "input_data_name": "dummy",
            "output_data_name": "dummy",
            "convolution_type": "integrate",
            "data_column_dict": {
                "delay_time": "delay_time",
                "normalized_yield": "probability",
            },
            "ignore_metallicity": True,
        }

        #
        self.convolution_config, data_dict, _ = extract_data(
            config=self.convolution_config,
            convolution_instruction=normal_convolution_instructions,
        )

        #
        sfr_dict = self.convolution_config["SFR_info"]

        time_bin_info_dict = {
            "bin_number": 0,
            "bin_center": 0.5 * u.yr,
            "bin_edge_lower": 0,
            "bin_size": 1 * u.yr,
            "bin_type": "convolution time",
            "time_type": self.convolution_config["time_type"],
        }

        #
        convolution_result = convolution_by_integration(
            time_bin_info_dict=time_bin_info_dict,
            sfr_dict=sfr_dict,
            config=self.convolution_config,
            convolution_instruction=normal_convolution_instructions,
            data_dict=data_dict,
        )

        #
        np.testing.assert_array_equal(
            convolution_result["convolution_results"]["yield"],
            np.array([1, 2, 3, 4.0]) * (1.0 / u.yr / u.Gpc**3),
        )

    def test_convolution_by_integration_post_convolution_simple(self):
        def simple_post_convolution_function(
            config,
            sfr_dict,
            data_dict,
            convolution_results,
            convolution_instruction,
        ):
            convolution_results["yield"] = convolution_results["yield"] * 0

            return convolution_results

        #
        normal_convolution_instructions = {
            **default_convolution_instruction,
            "input_data_name": "dummy",
            "output_data_name": "dummy",
            "convolution_type": "integrate",
            "data_column_dict": {
                "delay_time": "delay_time",
                "normalized_yield": "probability",
            },
            "ignore_metallicity": True,
            "post_convolution_function": simple_post_convolution_function,
        }

        #
        self.convolution_config, data_dict, _ = extract_data(
            config=self.convolution_config,
            convolution_instruction=normal_convolution_instructions,
        )

        sfr_dict = self.convolution_config["SFR_info"]

        time_bin_info_dict = {
            "bin_number": 0,
            "bin_center": 0.5 * u.yr,
            "bin_edge_lower": 0,
            "bin_size": 1 * u.yr,
            "bin_type": "convolution time",
            "time_type": self.convolution_config["time_type"],
        }

        #
        convolution_result = convolution_by_integration(
            time_bin_info_dict=time_bin_info_dict,
            sfr_dict=sfr_dict,
            config=self.convolution_config,
            convolution_instruction=normal_convolution_instructions,
            data_dict=data_dict,
        )

        yield_result = convolution_result["convolution_results"]["yield"]

        #
        np.testing.assert_array_equal(
            yield_result,
            np.zeros(self.dummy_data["probability"].shape) * (1.0 / u.yr / u.Gpc**3),
        )


if __name__ == "__main__":
    unittest.main()
