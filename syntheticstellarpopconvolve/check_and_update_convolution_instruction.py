"""
Functions to check and update the convolution instructions
"""

from syntheticstellarpopconvolve.general_functions import check_required


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


def check_convolution_instruction(convolution_instruction, config):
    """
    Function to check convolution instructions
    """

    # required for all
    check_required(
        config=convolution_instruction,
        required_list=["input_data_type", "input_data_name", "output_data_name"],
    )

    if convolution_instruction["convolution_type"] == "integrate":

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
                convolution_instruction=convolution_instruction,
                data_key="data_column_dict",
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
                convolution_instruction=convolution_instruction,
                data_key="data_layer_dict",
            )

        ###########
        # custom structure instructions
        elif convolution_instruction["input_data_type"] == "custom":
            # TODO:
            raise ValueError("Custom input data type not supported yet")

    elif convolution_instruction["convolution_type"] == "sample":
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
                # "IDs",
                "normalized_yield",
            ],
        )

        ###########
        # custom structure instructions
        if convolution_instruction["input_data_type"] != "event":
            # TODO:
            raise ValueError(
                "input data other than event-type data currently not supported when sampling"
            )

    else:
        raise ValueError(
            "convolution type {} unsupported".format(
                convolution_instruction["convolution_type"]
            )
        )


def check_and_update_convolution_instruction(convolution_instruction, config):
    """
    Function to check convolution instructions
    """

    # check
    check_convolution_instruction(
        convolution_instruction=convolution_instruction, config=config
    )


def check_and_update_convolution_instructions(config):
    """
    Main function to check the convolution instructions.
    """

    if config["convolution_instructions"]:
        for convolution_instruction in config["convolution_instructions"]:
            check_and_update_convolution_instruction(
                convolution_instruction=convolution_instruction, config=config
            )
    else:
        raise ValueError("Please provide at least one convolution intruction")
