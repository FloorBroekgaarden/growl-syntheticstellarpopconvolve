"""
Some general functions related to the convolution
"""

import inspect
import json
import logging
import os
import shutil
import tempfile
from inspect import isfunction

import astropy.units as u
import numpy as np
import psutil
from astropy.cosmology import Planck13 as cosmo  # Planck 2013
from scipy import interpolate

from syntheticstellarpopconvolve.convolution_calculate_birth_redshift_array import (
    calculate_origin_redshift_array,
)

logger = logging.getLogger(__name__)


def get_username():
    """
    Function to get the username of the user that spawned the current process
    """

    return psutil.Process().username()


def extract_arguments(func, arg_dict):
    """
    Function that extracts the entries in 'arg_dict' that are arguments to the function 'func'
    """

    # get various arg types
    signature = inspect.signature(func)
    all_args = inspect.getfullargspec(func).args
    args_with_defaults = [
        k
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    ]
    args_without_defaults = [arg for arg in all_args if arg not in args_with_defaults]

    # construct args
    args = {arg: arg_dict[arg] for arg in args_without_defaults}

    # check if kwonlyargs are also passed along
    args_for_args_with_defaults = {
        arg: arg_dict[arg] for arg in args_with_defaults if arg in arg_dict.keys()
    }

    # combine args
    combined_args = {**args, **args_for_args_with_defaults}

    return combined_args


class JsonCustomEncoder(json.JSONEncoder):
    """Support for data types that JSON default encoder
    does not do.

    This includes:

        * Numpy array or number
        * Complex number
        * Set
        * Bytes
        * astropy.UnitBase
        * astropy.Quantity

    Examples
    --------
    >>> import json
    >>> import numpy as np
    >>> from astropy.utils.misc import JsonCustomEncoder
    >>> json.dumps(np.arange(3), cls=JsonCustomEncoder)
    '[0, 1, 2]'

    copied from astropy and extended

    """

    def default(self, obj):  # DH0001
        import numpy as np
        from astropy import units as u

        if isinstance(obj, u.Quantity):
            return dict(value=obj.value, unit=obj.unit.to_string())
        if isinstance(obj, (np.number, np.ndarray)):
            return obj.tolist()
        elif isinstance(obj, complex):
            return [obj.real, obj.imag]
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, bytes):  # pragma: py3
            return obj.decode()
        elif isinstance(obj, interpolate.interp1d):
            return str(obj)
        elif isinstance(obj, (u.UnitBase, u.FunctionUnitBase)):
            if obj == u.dimensionless_unscaled:
                obj = "dimensionless_unit"
            else:
                return obj.to_string()
        elif isinstance(obj, type(logger)):
            return str(obj)
        elif isinstance(obj, type(cosmo)):
            return str(obj)
        elif isinstance(obj, type(cosmo)):
            return str(obj)
        elif isfunction(obj):
            return str(obj)

        return json.JSONEncoder.default(self, obj)


####
# General functions
def custom_json_serializer(obj):  # DH0001
    """
    Custom serialiser for binary_c to use when functions are present in the dictionary
    that we want to export.

    Function objects will be turned into str representations of themselves

    Args:
        obj: The object that might not be serialisable

    Returns:
        Either string representation of object if the object is a function, or the object itself
    """

    if isinstance(obj, u.Quantity):
        return obj.value
    elif isinstance(obj, interpolate.interp1d):
        return str(obj)
    # elif isinstance(o, t)
    return obj


def verbose_print(  # DH0001
    message: str, verbosity: int, minimal_verbosity: int
) -> None:
    """
    Function that decides whether to print a message based on the current verbosity
    and its minimum verbosity

    if verbosity is equal or higher than the minimum, then we print

    Args:
        message: message to print
        verbosity: current verbosity level
        minimal_verbosity: threshold verbosity above which to print
    """

    if verbosity >= minimal_verbosity:
        print(message)


def vb(message, verbosity, minimal_verbosity):  # DH0001
    """
    Shorthand for verbose_print
    """

    verbose_print(message, verbosity, minimal_verbosity)


def handle_extra_weights_function(
    config,
    convolution_time_bin_center,
    convolution_instruction,
    data_dict,
    output_shape,
):
    """
    Function to handle the calculation of a set of extra weights that will be applied to the systems / sub-ensemble

    TODO: move this function elsewhere
    """

    # set default
    extra_weights = np.ones(output_shape)

    # handle calculation extra weights
    if convolution_instruction.get("extra_weights_function", None) is not None:
        # Construct what parameters are available for the extra function
        available_parameters = {
            "config": config,
            "time_value": convolution_time_bin_center,
            "convolution_instruction": convolution_instruction,
            "data_dict": data_dict,
            **convolution_instruction.get(
                "extra_weights_function_additional_parameters", {}
            ),  #
        }

        # Make sure we extract the correct things from the available parameters
        extra_weights_function_args = extract_arguments(
            func=convolution_instruction["extra_weights_function"],
            arg_dict=available_parameters,
        )

        #
        config["logger"].debug(
            "Calculating extra weights using function {} and arguments {}".format(
                convolution_instruction["extra_weights_function"].__name__,
                extra_weights_function_args,
            )
        )

        # Call extra function and calculate extra weights (with something like detection probability)
        extra_weights = convolution_instruction["extra_weights_function"](
            **extra_weights_function_args
        )
        if extra_weights is None:
            raise ValueError(
                "The extra function did not return a correct set of extra weights"
            )

    if extra_weights.shape != output_shape:
        raise ValueError(
            "Desired output shape does not match the shape of the extra weights"
        )

    return extra_weights


def calculate_origin_time_array(config, data_dict, convolution_time_bin_center):
    """
    Function to calculate the origin time array

    TODO: more elsewhere
    """

    config["logger"].debug("Calculating origin-time array")

    # if convolution method and SFR is the in lookback time, then we can just subtract
    if config["time_type"] == "lookback_time":
        origin_time_array = (
            np.ones(data_dict["delay_time"].shape) * convolution_time_bin_center
            + data_dict["delay_time"]
        )
        config["logger"].debug(
            "Calculating origin-time array based on lookback_time: {}".format(
                origin_time_array
            )
        )

    if config["time_type"] == "redshift":
        origin_time_array = calculate_origin_redshift_array(
            config=config,
            convolution_redshift_value=convolution_time_bin_center,
            data_dict=data_dict,
        )
        config["logger"].debug(
            "Calculating origin-time array based on redshift: {}".format(
                origin_time_array
            )
        )

    return origin_time_array


def calculate_digitized_sfr_rates(
    config, convolution_time_bin_center, data_dict, sfr_dict
):
    """
    Function to calculate the digitized rates

    TODO: update docstring
    TODO: more elsewhere
    """

    ###########
    # calculate origin time
    origin_time_array = calculate_origin_time_array(
        config=config,
        data_dict=data_dict,
        convolution_time_bin_center=convolution_time_bin_center,
    )

    # Get indices for birth redshift
    config["logger"].debug("Calculating digitized origin-time indices")

    digitized_time_indices = (
        np.digitize(
            origin_time_array, bins=sfr_dict["padded_time_bin_edges"], right=False
        )
        - 1
    )

    #
    if "metallicity" in data_dict.keys():

        # Get indices for metallicity values
        config["logger"].debug("Calculating digitized metallicity indices")
        metallicity_indices = (
            np.digitize(
                data_dict["metallicity"],
                bins=config["padded_metallicity_bin_edges"],
                right=False,
            )
            - 1
        )

        # Calculate rates
        config["logger"].debug("Calculating metallicity weighted SFR rates")
        digitised_sfr_rates = sfr_dict[
            "padded_metallicity_weighted_starformation_array"
        ][metallicity_indices, digitized_time_indices]

    else:
        # use JUST the SFR, not the metallicity dependent one

        # Calculate rates
        config["logger"].debug("Calculating absolute SFR rates")
        digitised_sfr_rates = sfr_dict["padded_starformation_array"][
            digitized_time_indices
        ]

    return digitised_sfr_rates


def calculate_bincenters(array, convert="linear"):
    """
    Function to calculate bincenters

    TODO: allow other conversions
    """

    if convert == "linear":
        bincenters = (array[1:] + array[:-1]) / 2

    return bincenters


def calculate_edge_values(arr):
    """
    Function to calculate the edge values given a bunch of centers
    """

    #
    diff = np.diff(arr)
    edge_values = (arr[1:] + arr[:-1]) / 2

    #
    edge_values = np.insert(
        edge_values,
        0,
        edge_values[0] - diff[0],
        axis=0,
    )

    #
    edge_values = np.insert(
        edge_values,
        edge_values.shape[0],
        edge_values[-1] + diff[-1],
        axis=0,
    )

    return edge_values


def pad_function(array, left_val, right_val, relative_to_edge_val, axis=0):
    """
    Function to pad an array
    """

    # copy
    padded_array = array[:]

    # check if there are units involved
    try:
        unit = padded_array.unit

        left_val = left_val * unit
        right_val = right_val * unit
    except AttributeError:
        pass

    #
    if relative_to_edge_val:
        #
        padded_array = np.insert(
            padded_array,
            0,
            padded_array[axis] + left_val,
            axis=axis,
        )

        #
        padded_array = np.insert(
            padded_array,
            padded_array.shape[axis],
            padded_array[-1] + right_val,
            axis=axis,
        )
    else:

        #
        padded_array = np.insert(
            padded_array,
            0,
            left_val,
            axis=axis,
        )

        #
        padded_array = np.insert(
            padded_array,
            padded_array.shape[axis],
            right_val,
            axis=axis,
        )

    return padded_array


def generate_group_name(convolution_instruction, sfr_dict):
    """
    Function to generate the group name. Also provides layers
    """

    #
    elements = []

    if sfr_dict is None:
        sfr_dict = {}

    #
    if sfr_dict.get("name", None) is not None:
        elements.append(sfr_dict["name"])

    #
    elements.append(convolution_instruction["input_data_type"])
    elements.append(convolution_instruction["input_data_name"])
    elements.append(convolution_instruction["output_data_name"])

    # construct groupname
    groupname = "/".join(elements)

    return groupname, elements


def get_tmp_dir(config, convolution_instruction, sfr_dict=None):
    """
    Function to get tmp dir
    """

    #
    groupname, _ = generate_group_name(
        convolution_instruction=convolution_instruction, sfr_dict=sfr_dict
    )

    #
    tmp_dir = os.path.join(config["tmp_dir"], groupname)

    return tmp_dir


def handle_custom_scaling_or_conversion(config, data_layer_or_column_dict_entry, value):
    """
    Function that handles multiplying the key of the ensemble with some value or with some function
    """

    ###########
    # Handle logic of multiple steps
    if ("conversion_factor" in data_layer_or_column_dict_entry.keys()) and (
        "conversion_function" in data_layer_or_column_dict_entry.keys()
    ):
        raise ValueError(
            "We currently do not support both a conversion factor and a conversion function"
        )

    ###########
    # convert data by a function
    if "conversion_factor" in data_layer_or_column_dict_entry.keys():
        value = value * data_layer_or_column_dict_entry["conversion_factor"]

        #
        config["logger"].debug(
            "Applying conversion factor {} on data column {}".format(
                data_layer_or_column_dict_entry["conversion_factor"],
                data_layer_or_column_dict_entry,
            )
        )

    ###########
    # convert data by a function
    if "conversion_function" in data_layer_or_column_dict_entry.keys():
        value = data_layer_or_column_dict_entry["conversion_function"](value)

        #
        config["logger"].debug(
            "Applying conversion function {} on data column {}".format(
                data_layer_or_column_dict_entry["conversion_function"].__name__,
                data_layer_or_column_dict_entry,
            )
        )

    return value


def temp_dir(*child_dirs: str, clean_path=False) -> str:
    """
    Function to create directory within the TMP directory of the file system, starting with `/<TMP>/binary_c_python-<username>`

    Makes use of os.makedirs exist_ok which requires python 3.2+

    Args:
        *child_dirs: str input where each next input will be a child of the previous full_path. e.g. ``temp_dir('tests', 'grid')`` will become ``'/tmp/binary_c_python-<username>/tests/grid'``
        *clean_path (optional): Boolean to make sure that the directory is cleaned if it exists
    Returns:
        the path of a sub directory called binary_c_python in the TMP of the file system
    """

    tmp_dir = tempfile.gettempdir()
    username = get_username()
    full_path = os.path.join(tmp_dir, "binary_c_python-{}".format(username))

    # loop over the other paths if there are any:
    if child_dirs:
        for extra_dir in child_dirs:
            full_path = os.path.join(full_path, extra_dir)

    # Check if we need to clean the path
    if clean_path and os.path.isdir(full_path):
        shutil.rmtree(full_path)

    #
    os.makedirs(full_path, exist_ok=True)

    return full_path
