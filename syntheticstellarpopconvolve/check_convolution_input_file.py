"""
Function to check the input file
"""

import h5py


def check_convolution_input_file(config):
    """
    TODO
    """

    #
    config["logger"].debug("Checking convolution input file")

    #
    input_file = h5py.File(config["input_filename"], "r")

    # check if there is data in the file
    if "input_data" not in input_file.keys():
        raise ValueError("Please provide a 'input_data' group in the input hdf5file.")

    # TODO: loop over all the convolution_instructions to make sure the required data exists for them.

    # check if there is a config group and
    if "config" not in input_file.keys():
        raise ValueError("Please provide a 'config' group in the input hdf5file.")

    # check if the config group contains information about the population (i.e. binary_c(-python)) config.
    if "population" not in input_file["config"].keys():
        raise ValueError(
            "Please provide a 'config/population' dataset in the input hdf5file."
        )
