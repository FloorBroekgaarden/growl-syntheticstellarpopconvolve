"""
Function to copy the input file and
"""

import json
import os
import shutil

import h5py

from syntheticstellarpopconvolve.convolution_general_functions import JsonCustomEncoder


def prepare_output_file(config):
    """
    Function to prepare the output file, create some initial groups and store the configuration
    """

    #
    config["logger"].debug("Preparing output file")

    # Copy input file to output file
    if os.path.isfile(config["output_filename"]):
        os.remove(config["output_filename"])
    shutil.copy(config["input_filename"], config["output_filename"])

    with h5py.File(config["output_filename"], "a") as output_hdf5file:
        # Store convolution configuration in
        output_hdf5file["config"].create_dataset(
            "convolution", data=json.dumps(config, cls=JsonCustomEncoder)
        )
