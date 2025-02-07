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
from syntheticstellarpopconvolve.general_functions import calculate_bin_edges, temp_dir

# # load the data
# example_ensemble_filename = pkg_resources.resource_filename(
#     "syntheticstellarpopconvolve", "example_data/example_ensemble.json"
# )
# with open(example_ensemble_filename, "r") as f_ensemble:
#     ensemble = json.loads(f_ensemble.read())

# inflated_ensemble = convert_ensemble_to_dataframe(
#     ensemble_data=ensemble["ensemble"]['Xyield'],
#     verbose=False,
#     contains_named_layers=True,
# )

# # Convert to
# inflated_ensemble = inflated_ensemble.astype({'time': 'float', 'probability': 'float'})

# # Create time bin edges
# sorted_unique_time_centers = np.sort(inflated_ensemble['time'].unique())
# time_bin_edges = calculate_bin_edges(sorted_unique_time_centers)

# # turn the values to correct base
# inflated_ensemble['time'] = 10**inflated_ensemble['time']
# time_bin_edges = 10**time_bin_edges
# # TODO: turn probability into correct base (d/dlog10 t -> d/dt)

# # ensure sorting
# inflated_ensemble = inflated_ensemble.sort_values(by=['time'])


records = [
    {"time": 0.5, "value": 10, "probability": 1},
    {"time": 1.5, "probability": 2, "value": 20},
]

example_dataframe = pd.DataFrame.from_records(records)
print(example_dataframe)


sorted_unique_time_centers = np.sort(example_dataframe["time"].unique())
time_bin_edges = calculate_bin_edges(sorted_unique_time_centers)
# print(time_bin_edges)
# quit()

#
TMP_DIR = temp_dir("code", "convolve_stochastically", clean_path=True)

# create file
input_hdf5_filename = os.path.join(TMP_DIR, "input_hdf5.h5")
output_hdf5_filename = os.path.join(TMP_DIR, "output_hdf5.h5")
input_hdf5_file = h5py.File(input_hdf5_filename, "w")

# Create groups main
input_hdf5_file.create_group("input_data")
input_hdf5_file.create_group("config")

# close
input_hdf5_file.close()

# store the data frame in the hdf5file
example_dataframe.to_hdf(input_hdf5_filename, key="input_data/binned_example")

#
convolution_config = copy.copy(default_convolution_config)
convolution_config["input_filename"] = input_hdf5_filename
convolution_config["output_filename"] = output_hdf5_filename
convolution_config["tmp_dir"] = TMP_DIR


###
# convolution instructions
convolution_config["convolution_instructions"] = [
    {
        "convolution_type": "integrate",
        "input_data_name": "binned_example",
        "output_data_name": "binned_example",
        "contains_binned_data": True,
        "ignore_metallicity": True,
        "delay_time_data_bin_info_dict": {
            "delay_time_data_bin_edges": time_bin_edges * u.yr
        },
        "data_column_dict": {
            # required
            "normalized_yield": "probability",
            "delay_time": {"column_name": "time", "unit": u.yr},
        },
    },
]

#
convolution_config["time_type"] = "lookback_time"
convolution_config["convolution_lookback_time_bin_edges"] = np.arange(0, 2, 1) * u.yr


# construct the sfr-dict (NOTE: this uses absolute SFR, not metallicity dependent)
sfr_dict = {}
sfr_dict["lookback_time_bin_edges"] = np.arange(0, 10, 1) * u.yr

#
sfr_dict["starformation_rate_array"] = (
    np.ones(sfr_dict["lookback_time_bin_edges"].shape[0] - 1) * u.Msun / u.yr
)  # example of a constant star-formation rate. this could be anything of course.

# store
convolution_config["SFR_info"] = sfr_dict

# convolve
convolve(config=convolution_config)

print("finished convolution")
