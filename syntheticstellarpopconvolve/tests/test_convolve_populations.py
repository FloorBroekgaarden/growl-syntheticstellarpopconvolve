"""
Testcases for convolve_populations file
"""

import copy
import os
import unittest

import astropy.units as u
import numpy as np
import pandas as pd

from syntheticstellarpopconvolve import (
    default_convolution_config,
    default_convolution_instruction,
)
from syntheticstellarpopconvolve.check_and_prepare_output_file import (
    check_and_prepare_output_file,
)
from syntheticstellarpopconvolve.check_and_update_convolution_config import (
    check_and_update_convolution_config,
)
from syntheticstellarpopconvolve.convolve_populations import (
    extract_data,
    generate_data_dict,
)
from syntheticstellarpopconvolve.general_functions import (
    generate_boilerplate_outputfile,
    temp_dir,
)

TMP_DIR = temp_dir(
    "tests", "tests_convolution", "tests_convolve_populations", clean_path=True
)


class test_generate_data_dict(unittest.TestCase):
    def test_generate_data_dict_events(self):

        #
        output_hdf5_filename = os.path.join(TMP_DIR, "output_hdf5_sfr_only.h5")
        generate_boilerplate_outputfile(output_hdf5_filename)

        ##############
        # SET UP DATA
        self.dummy_data = {
            "delay_time": np.array([0, 1, 2, 3]),
            "probability": np.array([1, 2, 3, 4]),
        }
        dummy_df = pd.DataFrame.from_records(self.dummy_data)

        ##############
        # Store data in pandas
        dummy_df.to_hdf(output_hdf5_filename, key="input_data/{}".format("dummy"))

        #
        self.convolution_config = copy.copy(default_convolution_config)

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
            },
        ]

        #
        self.convolution_config["tmp_dir"] = os.path.join(TMP_DIR, "tmp")

        #
        check_and_prepare_output_file(config=self.convolution_config)

        #
        check_and_update_convolution_config(self.convolution_config)

        #
        normal_convolution_instructions = {
            **default_convolution_instruction,
            "input_data_name": "dummy",
            "output_data_name": "dummy",
            "data_column_dict": {
                "delay_time": "delay_time",
                "normalized_yield": "probability",
            },
        }

        _, data_dict, _ = generate_data_dict(
            config=self.convolution_config,
            convolution_instruction=normal_convolution_instructions,
        )

        #
        np.testing.assert_array_equal(
            data_dict["delay_time"], self.dummy_data["delay_time"] * u.yr
        )


if __name__ == "__main__":
    unittest.main()
