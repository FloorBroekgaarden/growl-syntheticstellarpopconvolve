"""
Testcases for convolve_ensembles file
"""

import copy
import os
import unittest

import astropy.units as u
import numpy as np

from syntheticstellarpopconvolve import convolution_options_defaults
from syntheticstellarpopconvolve.convolution_calculate_birth_redshift_array import (
    calculate_origin_redshift_array,
)
from syntheticstellarpopconvolve.convolution_general_functions import temp_dir
from syntheticstellarpopconvolve.prepare_redshift_interpolator import (
    prepare_redshift_interpolator,
)

TMP_DIR = temp_dir(
    "tests",
    "tests_convolution",
    "tests_convolution_calculate_birth_redshift_array",
    clean_path=True,
)


class test_calculate_origin_redshift_array(unittest.TestCase):
    def test_calculate_origin_redshift_array_all_good(self):
        #
        convolution_config = copy.copy(convolution_options_defaults)
        convolution_config["redshift_interpolator_data_output_filename"] = os.path.join(
            TMP_DIR, "interpolator_dict.p"
        )
        convolution_config = prepare_redshift_interpolator(convolution_config)
        convolution_config["time_type"] = "redshift"

        #
        origin_redshift_array = calculate_origin_redshift_array(
            config=convolution_config,
            data_dict={"delay_time": np.array([1, 2, 3]) * 1e9 * u.yr},
            convolution_redshift_value=0.5,
        )
        np.testing.assert_array_almost_equal(
            origin_redshift_array,
            np.array([0.6501032923316669, 0.8336451543045214, 1.0661079791875108]),
        )

    def test_calculate_origin_redshift_array_one_too_far(self):
        #
        convolution_config = copy.copy(convolution_options_defaults)
        convolution_config["redshift_interpolator_data_output_filename"] = os.path.join(
            TMP_DIR, "interpolator_dict.p"
        )
        convolution_config = prepare_redshift_interpolator(convolution_config)
        convolution_config["time_type"] = "redshift"

        #
        origin_redshift_array = calculate_origin_redshift_array(
            config=convolution_config,
            data_dict={"delay_time": np.array([1, 2, 3 * 1e9]) * 1e9 * u.yr},
            convolution_redshift_value=0.5,
        )
        np.testing.assert_array_almost_equal(
            origin_redshift_array,
            np.array([0.6501032923316669, 0.8336451543045214, -1]),
        )


if __name__ == "__main__":
    unittest.main()
