"""
Function to handle checking the convolution configuration

TODO: handle logic of SFR
"""

import astropy.units as u
import voluptuous as vol

from syntheticstellarpopconvolve.convolution_default_settings import (
    convolution_options_defaults_dict,
)


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


def check_sfr_dict(sfr_dict, requires_name, requires_metallicity_info, time_type):
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

        # check if has units
        if not sfr_dict["lookback_time_bin_edges"].unit == u.yr:
            raise ValueError(
                "Please express 'lookback_time_bin_edges' in units of years (u.yr)"
            )

    elif time_type == "redshift":
        if "redshift_bin_edges" not in sfr_dict:
            raise ValueError("redshift_bin_edges is required in the sfr dictionary")

    ##########
    # Check if the correct time bins are present
    if "starformation_array" not in sfr_dict:
        raise ValueError("starformation_array is required in the sfr dictionary")

    ##########
    # Check if the shape of the time_bin_edges is 1 smaller than the starformation array

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
        for key, value in convolution_options_defaults_dict.items()
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

        if not config["convolution_lookback_time_bin_edges"].unit == u.yr:
            raise ValueError(
                "Please express 'convolution_lookback_time_bin_edges' in units of years (u.yr)"
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
            check_sfr_dict(
                sfr_dict=config["SFR_info"],
                requires_name=False,
                requires_metallicity_info=requires_metallicity_info,
                time_type=time_type,
            )
        elif isinstance(config["SFR_info"], list):
            # check all sfr dicts
            for sfr_dict in config["SFR_info"]:
                check_sfr_dict(
                    sfr_dict=sfr_dict,
                    requires_name=True,
                    requires_metallicity_info=requires_metallicity_info,
                    time_type=time_type,
                )
    else:
        raise ValueError("No SFR info has been provided. Aborting")
