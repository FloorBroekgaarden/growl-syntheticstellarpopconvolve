"""
Some general functions related to the convolution
"""

import functools
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

from syntheticstellarpopconvolve.calculate_birth_redshift_array import (
    calculate_origin_redshift_array,
)

logger = logging.getLogger(__name__)

dimensionless_unit = u.m / u.m


def get_username():
    """
    Function to get the username of the user that spawned the current process
    """

    return psutil.Process().username()


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


def calculate_origin_time_array(config, data_dict, convolution_time_bin_center):
    """
    Function to calculate the origin time array

    TODO: move elsewhere
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
    elif config["time_type"] == "redshift":
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
    else:
        raise ValueError("Choice for time-type unknown. {}".format(config["time_type"]))

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

    # Handle whether we want to specify metallicity as well
    if "metallicity" in data_dict.keys():

        # Get indices for metallicity values
        config["logger"].debug("Calculating digitized metallicity indices")
        metallicity_indices = (
            np.digitize(
                data_dict["metallicity"],
                bins=sfr_dict["padded_metallicity_bin_edges"],
                right=False,
            )
            - 1
        )

        # Calculate rates
        config["logger"].debug("Calculating metallicity weighted SFR rates")
        digitised_sfr_rates = sfr_dict[
            "padded_metallicity_weighted_starformation_rate_array"
        ][metallicity_indices, digitized_time_indices]
    else:
        # use JUST the SFR, not the metallicity dependent one

        # Calculate rates
        config["logger"].debug("Calculating absolute SFR rates")
        digitised_sfr_rates = sfr_dict["padded_starformation_rate_array"][
            digitized_time_indices
        ]

    #

    # handle multiplication by bin-size
    # TODO: clean and handle implementation
    # TODO: make sure that padded_time_binsizes exists.
    if config["multiply_by_time_binsize"]:
        # get indices
        time_binsize_indices = (
            np.digitize(
                origin_time_array, bins=sfr_dict["padded_time_bin_edges"], right=False
            )
            - 1
        )

        # get time-binsizes
        time_binsizes = sfr_dict["padded_time_binsizes"]

        # update sfr_rates
        digitised_sfr_rates = digitised_sfr_rates * time_binsizes[time_binsize_indices]

    return digitised_sfr_rates


def calculate_bincenters(array, convert="linear"):
    """
    Function to calculate bincenters

    TODO: allow other conversions
    """

    if convert == "linear":
        bincenters = (array[1:] + array[:-1]) / 2
    else:
        raise ValueError(f"convert choice {convert} is unknown")

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
    full_path = os.path.join(tmp_dir, "sspc-{}".format(username))

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


def check_required(config, required_list):
    """
    Function to check if the keys in the required_list are present in the convolution_instruction dict
    """

    for key in required_list:
        if key not in config.keys():
            raise ValueError(
                "{} is required in the convolution_instruction".format(key)
            )


def is_time_unit(parameter):
    """
    Function to check if a parameter has time-units
    """

    try:
        parameter.to(u.yr)
        return True
    except u.core.UnitConversionError:
        return False
    except AttributeError:
        return False


def has_unit(parameter, fail_on_dimensionless=True):
    """
    Function to check if a parameter has any unit assigned to it
    """

    try:
        unit = parameter.unit

        if fail_on_dimensionless:
            dimensionless_unit = u.m / u.m
            if unit == dimensionless_unit:
                return False
        return True
    except:
        return False


has_unit_dimensionless_okay = functools.partial(has_unit, fail_on_dimensionless=False)
