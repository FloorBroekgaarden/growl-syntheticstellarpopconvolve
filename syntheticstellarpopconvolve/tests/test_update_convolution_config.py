"""
Testcases for update_convolution_config file
"""

import copy
import os
import unittest

import astropy.units as u
import numpy as np

from syntheticstellarpopconvolve import convolution_options_defaults
from syntheticstellarpopconvolve.convolution_check_config import (
    check_convolution_config,
)
from syntheticstellarpopconvolve.convolution_general_functions import temp_dir
from syntheticstellarpopconvolve.update_convolution_config import (
    update_convolution_config,
)

TMP_DIR = temp_dir(
    "tests", "tests_convolution", "tests_update_convolution_config", clean_path=True
)


class test_update_convolution_config(unittest.TestCase):
    """ """

    def test_update_convolution_config(self):

        config = copy.copy(convolution_options_defaults)
        config["redshift_interpolator_data_output_filename"] = os.path.join(
            TMP_DIR, "interpolator_dict.p"
        )
        #
        config["input_filename"] = os.path.join(TMP_DIR, "input_hdf5_sfr_only.h5")
        config["output_filename"] = os.path.join(TMP_DIR, "output_hdf5_sfr_only.h5")

        # Set up SFR
        config["SFR_info"] = {
            "redshift_bin_edges": np.array([0, 1, 2, 3, 4, 5]),
            "starformation_array": np.array([1, 1, 1, 1, 1]) * u.Msun / u.yr / u.Gpc**3,
        }

        #
        config["convolution_instructions"] = [
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

        with open(config["input_filename"], "a") as _:
            pass

        config["convolution_redshift_bin_edges"] = np.array([1, 2])

        #
        check_convolution_config(config=config)

        #
        config = update_convolution_config(config=config)

        #
        self.assertEqual(config["convolution_time_bin_centers"].tolist(), [1.5])


if __name__ == "__main__":
    unittest.main()
