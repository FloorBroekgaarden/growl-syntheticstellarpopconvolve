"""
example script
"""

import copy
import json
import os
import time

import astropy.units as u
import h5py
import numpy as np
import pandas as pd
import pkg_resources

from syntheticstellarpopconvolve import convolve, default_convolution_config
from syntheticstellarpopconvolve.convolve_stochastically import (
    select_dict_entries_with_new_indices,
)
from syntheticstellarpopconvolve.ensemble_utils import convert_ensemble_to_dataframe
from syntheticstellarpopconvolve.general_functions import temp_dir

# load the data
example_ensemble_filename = pkg_resources.resource_filename(
    "syntheticstellarpopconvolve", "example_data/example_ensemble.json"
)
with open(example_ensemble_filename, "r") as f_ensemble:
    ensemble = json.loads(f_ensemble.read())


inflated_ensemble = convert_ensemble_to_dataframe(
    ensemble_data=ensemble["ensemble"]["Xyield"],
    verbose=False,
    contains_named_layers=True,
)

print(inflated_ensemble.head())


quit()


#
TMP_DIR = temp_dir("code", "convolve_stochastically", clean_path=True)

# create file
input_hdf5_filename = os.path.join(TMP_DIR, "input_hdf5.h5")
output_hdf5_filename = os.path.join(TMP_DIR, "output_hdf5.h5")
input_hdf5_file = h5py.File(input_hdf5_filename, "w")

# Create groups main
input_hdf5_file.create_group("input_data")
input_hdf5_file.create_group("config")

# add group for events
input_hdf5_file.create_group("input_data/events")

# Write population config to file
input_hdf5_file.create_dataset("config/population", data=json.dumps({}))

# close
input_hdf5_file.close()

# store the data frame in the hdf5file
wd_binaries.to_hdf(input_hdf5_filename, key="input_data/events/stochastic_example")

#
convolution_config = copy.copy(default_convolution_config)
convolution_config["input_filename"] = input_hdf5_filename
convolution_config["output_filename"] = output_hdf5_filename
convolution_config["tmp_dir"] = TMP_DIR
convolution_config["redshift_interpolator_data_output_filename"] = os.path.join(
    TMP_DIR, "interpolator_dict.p"
)
convolution_config["multiply_by_time_binsize"] = False
convolution_config["filter_future_events"] = False


###
# convolution instructions
convolution_config["convolution_instructions"] = [
    {
        "convolution_type": "sample",
        "input_data_name": "stochastic_example",
        "output_data_name": "stochastic_example",
        "ignore_metallicity": True,
        "filter_future_events": False,
        "post_convolution_function": post_convolution_function,
        "data_column_dict": {
            # required
            "normalized_yield": "normalized_yield",
            "delay_time": {"column_name": "time", "unit": u.Myr},
        },
    },
]

#
convolution_config["time_type"] = "lookback_time"
convolution_config["convolution_lookback_time_bin_edges"] = np.arange(3, 6, 1) * u.Gyr
print(convolution_config["convolution_lookback_time_bin_edges"])
quit()

# construct the sfr-dict (NOTE: this uses absolute SFR, not metallicity dependent)
sfr_dict = {}
sfr_dict["lookback_time_bin_edges"] = (np.arange(0, 10, 1) * u.Gyr).to(u.yr)

#
scale = 1e-5
sfr_dict["starformation_rate_array"] = (
    scale * np.ones(sfr_dict["lookback_time_bin_edges"].shape[0] - 1) * u.Msun / u.yr
)  # example of a constant star-formation rate. this could be anything of course.

# store
convolution_config["SFR_info"] = sfr_dict

# convolve
convolve(config=convolution_config)

print("finished convolution")


# read out content and integrate until today
with h5py.File(convolution_config["output_filename"], "r") as output_hdf5_file:

    print(output_hdf5_file.keys())
    print(output_hdf5_file["output_data"].keys())
    print(output_hdf5_file["output_data/event"].keys())
    print(output_hdf5_file["output_data/event/stochastic_example"].keys())
    print(
        output_hdf5_file[
            "output_data/event/stochastic_example/stochastic_example"
        ].keys()
    )
    # print(
    #     output_hdf5_file[
    #         "output_data/event/stochastic_example/stochastic_example/convolution_results/1500000000.0 yr"
    #     ].keys()
    # )

    print(
        output_hdf5_file[
            "output_data/event/stochastic_example/stochastic_example/convolution_results/set_1"
        ].keys()
    )
    print(
        output_hdf5_file[
            "output_data/event/stochastic_example/stochastic_example/convolution_results/set_2"
        ].keys()
    )

    quit()

    print(
        output_hdf5_file[
            "output_data/event/stochastic_example/stochastic_example/convolution_results"
        ].keys()
    )

    formation_time_bin_keys = list(
        output_hdf5_file[
            "output_data/event/stochastic_example/stochastic_example/convolution_results"
        ].keys()
    )

    ################
    #

    # loop over the formation-time bins
    formation_time_bin_keys = sorted(
        formation_time_bin_keys, key=lambda x: float(x.split(" ")[0])
    )
    for formation_time_bin_key in formation_time_bin_keys:

        # formation_time_bin_key = "3500000000.0 yr"
        print("=================================")
        print(f"formation_time_bin_key: {formation_time_bin_key}")
        print("=================================")

        ###########
        # Read out data

        # convert units
        unit_dict = json.loads(
            output_hdf5_file[
                f"output_data/event/stochastic_example/stochastic_example/convolution_results/{formation_time_bin_key}"
            ].attrs["units"]
        )
        unit_dict = {key: u.Unit(val) for key, val in unit_dict.items()}
        print(unit_dict)


#         indices = output_hdf5_file[
#             f"output_data/event/stochastic_example/stochastic_example/convolution_results/{formation_time_bin_key}/indices"
#         ][()]
#         # print(indices)

#         print(type(indices))
