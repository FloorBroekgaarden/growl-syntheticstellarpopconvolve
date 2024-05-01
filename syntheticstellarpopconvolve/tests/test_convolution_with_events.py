"""
Testcases for convolution_with_events file

TODO: test things with redshift
TODO: test things with multiply SFR histories
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

from syntheticstellarpopconvolve import convolution_options_defaults, convolve
from syntheticstellarpopconvolve.convolution_general_functions import temp_dir

TMP_DIR = temp_dir(
    "tests", "tests_convolution", "test_convolution_with_events", clean_path=True
)


class test_convolution_with_events(unittest.TestCase):
    """ """

    def test_convolution_with_events_with_lookback_time(self):
        """ """

        #
        input_hdf5_filename = os.path.join(TMP_DIR, "input_hdf5_sfr_only.h5")
        output_hdf5_filename = os.path.join(TMP_DIR, "output_hdf5_sfr_only.h5")

        ##############
        # SET UP DATA
        dummy_data = {
            "delay_time": np.array([0, 1, 2, 3]),
            "probability": np.array([1, 2, 3, 4]),
        }
        dummy_df = pd.DataFrame.from_records(dummy_data)

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
        convolution_config = copy.copy(convolution_options_defaults)

        # Set up SFR
        convolution_config["SFR_info"] = {
            "lookback_time_bin_edges": np.array([0, 1, 2, 3, 4, 5]) * u.yr,
            "starformation_array": np.array([1, 1, 1, 1, 1]) * u.Msun / u.yr / u.Gpc**3,
        }

        # set up convolution bins
        convolution_config["convolution_lookback_time_bin_edges"] = (
            np.array([0, 1, 2, 3, 4]) * u.yr
        )

        # lookback time convolution only
        convolution_config["time_type"] = "lookback_time"

        #
        convolution_config["input_filename"] = input_hdf5_filename
        convolution_config["output_filename"] = output_hdf5_filename

        convolution_config["redshift_interpolator_data_output_filename"] = os.path.join(
            TMP_DIR, "interpolator_dict.p"
        )

        #
        convolution_config["convolution_instructions"] = [
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
        # convolution_config["logger"].setLevel("DEBUG")

        #
        convolution_config["tmp_dir"] = os.path.join(TMP_DIR, "tmp")

        #
        convolve(config=convolution_config)

        #
        with h5py.File(output_hdf5_filename, "r") as output_hdf5_file:

            #
            arr_ = output_hdf5_file[
                "output_data/event/dummy/dummy/convolved_array/0.5 yr"
            ][()]
            self.assertTrue(np.array_equal(arr_, np.array([1, 2, 3, 4])))

            #
            arr_ = output_hdf5_file[
                "output_data/event/dummy/dummy/convolved_array/1.5 yr"
            ][()]
            self.assertTrue(np.array_equal(arr_, np.array([1, 2, 3, 4])))

            #
            arr_ = output_hdf5_file[
                "output_data/event/dummy/dummy/convolved_array/2.5 yr"
            ][()]
            self.assertTrue(np.array_equal(arr_, np.array([1, 2, 3, 0])))

            #
            arr_ = output_hdf5_file[
                "output_data/event/dummy/dummy/convolved_array/3.5 yr"
            ][()]
            self.assertTrue(np.array_equal(arr_, np.array([1, 2, 0, 0])))

            #
            self.assertTrue("SFR_info" in output_hdf5_file["output_data"].attrs.keys())


if __name__ == "__main__":
    unittest.main()
