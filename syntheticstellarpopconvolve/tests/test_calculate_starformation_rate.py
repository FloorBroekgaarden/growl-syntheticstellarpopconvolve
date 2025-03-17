"""
This is the unittest file for the calculate_starformation_rate.py source file

TODO: binned data backward integrate absolute
TODO: binned data backward integrate metallicity-weighted

TODO: nonbinned data backward integrate absolute
TODO: nonbinned data backward integrate metallicity-weighted

TODO: forward integrate absolute
TODO: forward integrate metallicity-weighted
"""

import copy
import logging
import os
import unittest

import astropy.units as u
import numpy as np

from syntheticstellarpopconvolve import (
    default_convolution_config,
    default_convolution_instruction,
)
from syntheticstellarpopconvolve.calculate_starformation_rate import (
    calculate_digitized_sfr_rates_non_binned_data_for_backward_convolution,
)
from syntheticstellarpopconvolve.check_and_update_sfr_dict import (
    check_and_update_sfr_dict,
)
from syntheticstellarpopconvolve.general_functions import temp_dir

np.random.seed(0)

TMP_DIR = temp_dir(
    "tests",
    "tests_convolution",
    "tests_calculat_starformation_rate",
    clean_path=True,
)


class test_calculate_starformation_rate_nonbinned_data_for_backward_convolution(
    unittest.TestCase
):

    def test_calculate_starformation_rate_nonbinned_data_for_backward_convolution_absolute(
        self,
    ):

        ##############
        self.dummy_data = {
            "delay_time": np.array([0, 1, 2, 3]) * u.yr,
            "value": np.array([3, 2, 1, 0]),
            "probability": np.array([1, 2, 3, 4]),
        }

        ##############
        self.dummy_data_with_metallicity = {
            "delay_time": np.array([0, 1, 2, 3, 0, 1, 2, 3]) * u.yr,
            "value": np.array([3, 2, 1, 0, 3, 2, 1, 0]),
            "probability": np.array([1, 2, 3, 4, 1, 2, 3, 4]),
            "metallicity": np.array([0.25, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75]),
        }

        # Set up SFR
        self.convolution_config = copy.copy(default_convolution_config)
        self.convolution_config["SFR_info"] = {
            "lookback_time_bin_edges": np.array([0, 1, 2, 3, 4, 5]) * u.yr,
            "starformation_rate_array": np.array([1, 2, 3, 4, 5]) * u.Msun / u.yr,
            "metallicity_bin_edges": np.array([0.0, 0.5, 1.0]),
            "metallicity_distribution_array": np.array(
                [[0.4, 0.6, 0.8, 5, 1.2], [1.6, 1.4, 1.2, 15, 0.8]]
            ).T,
        }
        self.convolution_config["convolution_lookback_time_bin_edges"] = (
            np.array([0, 1]) * u.yr
        )
        self.convolution_config["logger"].setLevel(logging.CRITICAL)
        self.convolution_config["tmp_dir"] = os.path.join(TMP_DIR, "tmp")
        self.convolution_config["SFR_info"] = check_and_update_sfr_dict(
            sfr_dict=self.convolution_config["SFR_info"],
            requires_name=False,
            requires_metallicity_info=True,
            time_type=self.convolution_config["time_type"],
            config=self.convolution_config,
        )

        #########
        #
        data_dict = self.dummy_data
        config = self.convolution_config
        convolution_instruction = {
            **default_convolution_instruction,
            "input_data_name": "dummy",
            "output_data_name": "dummy",
            "convolution_direction": "backward",
            "convolution_type": "integrate",
            "data_column_dict": {
                "delay_time": "delay_time",
                "normalized_yield": "probability",
            },
        }
        sfr_dict = self.convolution_config["SFR_info"]
        convolution_lookback_time_bin_edges = self.convolution_config[
            "convolution_lookback_time_bin_edges"
        ]
        convolution_time_bin_center = (
            (
                convolution_lookback_time_bin_edges[1:]
                + convolution_lookback_time_bin_edges[:-1]
            )
            / 2
        )[0]

        #
        starformation = (
            calculate_digitized_sfr_rates_non_binned_data_for_backward_convolution(
                config=config,
                convolution_instruction=convolution_instruction,
                convolution_time_bin_center=convolution_time_bin_center,
                data_dict=data_dict,
                sfr_dict=sfr_dict,
            )
        )

        print(starformation)


if __name__ == "__main__":
    unittest.main()
