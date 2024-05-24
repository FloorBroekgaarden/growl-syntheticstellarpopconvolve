"""
Function to handle checking the convolution configuration

TODO: handle logic of SFR
"""

import astropy.units as u
import numpy as np
import voluptuous as vol

from syntheticstellarpopconvolve.cosmology_utils import redshift_to_lookback_time
from syntheticstellarpopconvolve.default_convolution_config import (
    default_convolution_config_dict,
)
from syntheticstellarpopconvolve.general_functions import pad_function
from syntheticstellarpopconvolve.store_redshift_shell_info import (
    store_redshift_shell_info,
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


def check_metallicity(convolution_instruction, data_key):
    """
    Function to check the metallicity
    """

    if "ignore_metallicity" not in convolution_instruction.keys():
        if "metallicity" not in convolution_instruction.get(data_key, {}).keys():
            if "metallicity_value" not in convolution_instruction.keys():
                raise ValueError(
                    "If no metallicity value column / layer is provided, you either need to give 'metallicity_value' or set 'ignore_metallicity' to True"
                )


def update_sfr_dict(sfr_dict, config):
    """
    Function to update the SFR dict
    - provides padding
    - adds redshift shell info
    """

    #
    config["logger"].debug("Updating SFR dict")

    # Pad the SFR dict with the empty bins around
    sfr_dict = pad_sfr_dict(config=config, sfr_dict=sfr_dict)

    # Add redshift shell info to dict.
    sfr_dict = store_redshift_shell_info(config=config, sfr_dict=sfr_dict)

    return sfr_dict


def pad_sfr_dict(config, sfr_dict):
    """
    Function to pad the entries in the sfr dictionary with empty bins.

    These functions update all the sfr properties and adds new entries that are prepended with 'padded_'

    TODO: add time binsizes
    """

    #
    config["logger"].debug("Padding SFR dict")

    max_pad = 1.0e13

    ##########
    # pad lookback time/redshift array
    if config["time_type"] == "lookback_time":
        # pad lookback time bins
        sfr_dict["padded_lookback_time_bin_edges"] = pad_function(
            array=sfr_dict["lookback_time_bin_edges"],
            left_val=-max_pad,
            right_val=max_pad,
            relative_to_edge_val=True,
        )

        #
        sfr_dict["padded_time_bin_edges"] = sfr_dict["padded_lookback_time_bin_edges"]
        sfr_dict["time_bin_edges"] = sfr_dict["lookback_time_bin_edges"]

        #
        config["logger"].debug(
            "Padded lookback time bin edges {} to {}".format(
                sfr_dict["lookback_time_bin_edges"],
                sfr_dict["padded_lookback_time_bin_edges"],
            )
        )

        #
        sfr_dict["lookback_time_bin_sizes"] = np.abs(
            np.diff(sfr_dict["lookback_time_bin_edges"])
        )
        sfr_dict["time_bin_sizes"] = sfr_dict["lookback_time_bin_sizes"]

        # Pad time-bin sizes
        sfr_dict["padded_redshift_bin_sizeslookback_time_bin_sizes"] = pad_function(
            array=sfr_dict["lookback_time_bin_sizes"],
            left_val=0,
            right_val=0,
            relative_to_edge_val=False,
        )

        # log the binsizes
        config["logger"].debug(
            "Created lookback time bin sizes {}".format(
                sfr_dict["lookback_time_bin_sizes"],
            )
        )

    elif config["time_type"] == "redshift":
        #
        sfr_dict["padded_redshift_bin_edges"] = pad_function(
            array=sfr_dict["redshift_bin_edges"],
            left_val=-max_pad,
            right_val=max_pad,
            relative_to_edge_val=True,
        )

        #
        sfr_dict["padded_time_bin_edges"] = sfr_dict["padded_redshift_bin_edges"]
        sfr_dict["time_bin_edges"] = sfr_dict["redshift_bin_edges"]

        #
        config["logger"].debug(
            "Padded redshift bin edges {} to {}".format(
                sfr_dict["redshift_bin_edges"],
                sfr_dict["padded_redshift_bin_edges"],
            )
        )

        # create redshift time-bin size
        sfr_dict["redshift_bin_sizes"] = np.abs(np.diff(sfr_dict["redshift_bin_edges"]))

        # convert into actual time-bin sizes
        lookback_time_at_redshift_bin_edges = np.array(
            [
                redshift_to_lookback_time(
                    redshift=redshift_bin_edge, cosmology=config["cosmology"]
                )
                for redshift_bin_edge in sfr_dict["redshift_bin_edges"]
            ]
        )
        sfr_dict["time_bin_sizes"] = np.abs(
            np.diff(lookback_time_at_redshift_bin_edges)
        )

        # Pad time-bin sizes
        sfr_dict["padded_redshift_bin_sizes"] = pad_function(
            array=sfr_dict["redshift_bin_sizes"],
            left_val=0,
            right_val=0,
            relative_to_edge_val=False,
        )

        # log the binsizes
        config["logger"].debug(
            "Created redshift bin sizes {} and the corresponding lookback time-bin sizes {} ".format(
                sfr_dict["redshift_time_bin_sizes"], sfr_dict["time_bin_sizes"]
            )
        )
    else:
        raise ValueError("Invalid time-type")

    #########
    # Pad time-bin sizes
    sfr_dict["padded_time_bin_sizes"] = pad_function(
        array=sfr_dict["time_bin_sizes"],
        left_val=0,
        right_val=0,
        relative_to_edge_val=False,
    )

    # log the binsizes
    config["logger"].debug(
        "Padded the time bin sizes {}".format(
            sfr_dict["padded_time_bin_sizes"],
        )
    )

    ##########
    # pad SFR rate array
    if "starformation_array" in sfr_dict:  # it should be present always
        #
        sfr_dict["padded_starformation_array"] = pad_function(
            array=sfr_dict["starformation_array"],
            left_val=0,
            right_val=0,
            relative_to_edge_val=False,
        )

        #
        config["logger"].debug(
            "Padded starformation array {} to {}".format(
                sfr_dict["starformation_array"],
                sfr_dict["padded_starformation_array"],
            )
        )

    ##########
    # pad metallicity bins
    if "metallicity_bin_edges" in sfr_dict:
        #
        sfr_dict["padded_metallicity_bin_edges"] = pad_function(
            array=sfr_dict["metallicity_bin_edges"],
            left_val=1e-20,
            right_val=1,
            relative_to_edge_val=False,
        )

        #
        config["logger"].debug(
            "Padded metallicity bin edges {} to {}".format(
                sfr_dict["metallicity_bin_edges"],
                sfr_dict["padded_metallicity_bin_edges"],
            )
        )

    ##########
    # pad metallicity weighted SFR rate bins
    if "metallicity_weighted_starformation_array" in sfr_dict:
        #
        sfr_dict["padded_metallicity_weighted_starformation_array"] = pad_function(
            array=sfr_dict["metallicity_weighted_starformation_array"],
            left_val=0,
            right_val=0,
            relative_to_edge_val=False,
        )

        #
        sfr_dict["padded_metallicity_weighted_starformation_array"] = pad_function(
            array=sfr_dict["padded_metallicity_weighted_starformation_array"],
            left_val=0,
            right_val=0,
            relative_to_edge_val=False,
            axis=1,
        )

    return sfr_dict


def check_sfr_dict(
    sfr_dict, config, requires_name, requires_metallicity_info, time_type
):
    """
    Function to check the sfr dictionary
    """

    ##########
    # Check if the name exists if the sfr dict requires it
    if requires_name:
        if "name" not in sfr_dict:
            raise ValueError("Name is required in the sfr dictionary")

    ##########
    # Check if the correct time bins are present
    if time_type == "lookback_time":
        if "lookback_time_bin_edges" not in sfr_dict:
            raise ValueError(
                "lookback_time_bin_edges is required in the sfr dictionary"
            )

        # check if has time-units
        if not is_time_unit(sfr_dict["lookback_time_bin_edges"]):
            raise ValueError(
                "Please express 'lookback_time_bin_edges' in units of time"
            )

    elif time_type == "redshift":
        if "redshift_bin_edges" not in sfr_dict:
            raise ValueError("redshift_bin_edges is required in the sfr dictionary")

    ##########
    # Check if the correct time bins are present
    if "starformation_array" not in sfr_dict:
        raise ValueError("starformation_array is required in the sfr dictionary")

    # check if starformation array has any unit
    try:
        sfr_dict["starformation_array"].unit
    except AttributeError:
        raise AttributeError("starformation_array requires an astropy unit")

    ##########
    # TODO: Check if the shape of the time_bin_edges is 1 smaller than the starformation array

    ##########
    # check if metallicity information is present
    if requires_metallicity_info:
        # Check if the metallicity bins are present
        if "metallicity_bin_edges" not in sfr_dict:
            raise ValueError("metallicity_bin_edges is required in the sfr dictionary")

        # check if the MSSFR is present
        if "metallicity_weighted_starformation_array" not in sfr_dict:
            raise ValueError(
                "metallicity_weighted_starformation_array is required in the sfr dictionary"
            )

        # check if starformation array has any unit
        try:
            sfr_dict["metallicity_weighted_starformation_array"].unit
        except AttributeError:
            raise AttributeError(
                "metallicity_weighted_starformation_array requires an astropy unit"
            )

    ##########
    # update the SFR dict with extra things
    sfr_dict = update_sfr_dict(sfr_dict=sfr_dict, config=config)

    return sfr_dict


def check_required(config, required_list):
    """
    Function to check if the keys in the required_list are present in the convolution_instruction dict
    """

    for key in required_list:
        if key not in config.keys():
            raise ValueError(
                "{} is required in the convolution_instruction".format(key)
            )


def check_convolution_instruction(convolution_instruction):
    """
    Function to check convolution instructions
    """

    # required for all
    check_required(
        config=convolution_instruction,
        required_list=["input_data_type", "input_data_name", "output_data_name"],
    )

    ################
    # check event-specific instructions
    if convolution_instruction["input_data_type"] == "event":

        check_required(
            config=convolution_instruction,
            required_list=[
                "data_column_dict",
            ],
        )

        #
        check_required(
            config=convolution_instruction["data_column_dict"],
            required_list=[
                "delay_time",
                "yield_rate",
            ],
        )

        # check how metallicity is treated
        check_metallicity(
            convolution_instruction=convolution_instruction, data_key="data_column_dict"
        )

        # TODO: if a second function is passed along (to calculate the
        # detectability for example), then lets check if the user also provided
        # a dictionary that links the function parameter name to the column name
        # of the correct pandas table.

    ################
    # check ensemble-specific instructions
    elif convolution_instruction["input_data_type"] == "ensemble":

        # data
        check_required(
            config=convolution_instruction,
            required_list=[
                "data_layer_dict",
            ],
        )

        # the data layer dict requires only to have the delay time layer. the yield rate layer iks implied to be the deepest one
        check_required(
            config=convolution_instruction["data_layer_dict"],
            required_list=[
                "delay_time",
            ],
        )

        # check how metallicity is treated
        check_metallicity(
            convolution_instruction=convolution_instruction, data_key="data_layer_dict"
        )

    ###########
    # custom structure instructions
    elif convolution_instruction["input_data_type"] == "custom":
        # TODO:
        pass


def check_convolution_config(config):
    """
    Function to handle checking the convolution config
    """

    #
    config["logger"].debug("Checking configuration")

    ##########
    # Skip the convolution
    if not config["check_convolution_config"]:
        return

    ##########
    # from the main dictionary, create a validation scheme
    validation_dict = {
        key: value["validation"]
        for key, value in default_convolution_config_dict.items()
        if "validation" in value
    }
    validation_schema = vol.Schema(validation_dict, extra=vol.ALLOW_EXTRA)

    ##########
    # do the validation: some parameters require others to be set,
    for parameter, parameter_dict in config.items():
        # #
        # if parameter == "SFR_file":
        #     if not config["use_SFR_file"]:
        #         continue

        #
        if parameter == "redshift_interpolator_data_output_filename":
            if config["time_type"] != "redshift":
                continue
        if parameter in [
            "convolution_redshift_bin_edges",
            "convolution_lookback_time_bin_edges",
        ]:
            continue

        #
        validation_schema({parameter: parameter_dict})

    ##########
    # Perform other custom checks

    #######
    # check the convolution instructions
    if config["convolution_instructions"]:
        for convolution_instruction in config["convolution_instructions"]:
            check_convolution_instruction(
                convolution_instruction=convolution_instruction
            )
    else:
        raise ValueError("Please provide at least one convolution intruction")

    # determine whether any of the convolution instructions require metallicity
    requires_metallicity_info = any(
        [
            not convolution_instruction.get("ignore_metallicity", False)
            for convolution_instruction in config["convolution_instructions"]
        ]
    )

    # extract time-type from general config
    time_type = config["time_type"]

    ######
    # Check the convolution time or redshift
    if time_type == "lookback_time":
        # check if that is present
        if config.get("convolution_lookback_time_bin_edges", None) is None:
            raise ValueError(
                "Please provide 'convolution_lookback_time_bin_edges' when using 'lookback-time' as 'time-type'"
            )

        if not is_time_unit(config["convolution_lookback_time_bin_edges"]):
            # if not config["convolution_lookback_time_bin_edges"].unit == u.yr:
            raise ValueError(
                "Please express 'convolution_lookback_time_bin_edges' in units of time"
            )

        config["convolution_time_bin_edges"] = config[
            "convolution_lookback_time_bin_edges"
        ]
    elif time_type == "redshift":
        if config.get("convolution_redshift_bin_edges", None) is None:
            raise ValueError(
                "Please provide 'convolution_redshift_bin_edges' when using 'redshift' as 'time-type'"
            )
        config["convolution_time_bin_edges"] = config["convolution_redshift_bin_edges"]
    else:
        raise ValueError("unsupported time-type")

    #######
    # check the SFR information
    if "SFR_info" in config:
        if isinstance(config["SFR_info"], dict):
            config["SFR_info"] = check_sfr_dict(
                sfr_dict=config["SFR_info"],
                requires_name=False,
                requires_metallicity_info=requires_metallicity_info,
                time_type=time_type,
                config=config,
            )
        elif isinstance(config["SFR_info"], list):
            # check all sfr dicts
            for sfr_dict in config["SFR_info"]:
                sfr_dict = check_sfr_dict(
                    sfr_dict=sfr_dict,
                    requires_name=True,
                    requires_metallicity_info=requires_metallicity_info,
                    time_type=time_type,
                    config=config,
                )
    else:
        raise ValueError("No SFR info has been provided. Aborting")
