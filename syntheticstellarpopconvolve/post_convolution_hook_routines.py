"""
File containing methods to support post-convolution hook functionality
"""

import inspect


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


# def handle_extra_weights_function(
#     config,
#     bin_center,
#     convolution_instruction,
#     sfr_dict,
#     data_dict,
#     output_shape,
# ):
#     """
#     Function to handle the calculation of a set of extra weights that
#     will be applied to the systems / sub-ensemble

#     TODO: function calls like this can be generalized
#     """

#     # set default
#     extra_weights = np.ones(output_shape)

#     # handle calculation extra weights
#     if convolution_instruction.get("extra_weights_function", None) is not None:
#         # Construct what parameters are available for the extra function
#         available_parameters = {
#             "config": config,
#             "time_value": bin_center,
#             "convolution_instruction": convolution_instruction,
#             "sfr_dict": sfr_dict,
#             "data_dict": data_dict,
#             **convolution_instruction.get(
#                 "extra_weights_function_additional_parameters", {}
#             ),  #
#         }

#         # Make sure we extract the correct things from the available parameters
#         extra_weights_function_args = extract_arguments(
#             func=convolution_instruction["extra_weights_function"],
#             arg_dict=available_parameters,
#         )

#         #
#         config["logger"].debug(
#             "Calculating extra weights using function {} and arguments {}".format(
#                 convolution_instruction["extra_weights_function"].__name__,
#                 extra_weights_function_args,
#             )
#         )

#         # Call extra function and calculate extra weights (with something like detection probability)
#         extra_weights = convolution_instruction["extra_weights_function"](
#             **extra_weights_function_args
#         )
#         if extra_weights is None:
#             raise ValueError(
#                 "The extra function did not return a correct set of extra weights"
#             )

#     if extra_weights.shape != output_shape:
#         raise ValueError(
#             "Desired output shape does not match the shape of the extra weights"
#         )

#     return extra_weights


# def handle_position_sampling_function(
#     config,
#     job_dict,
#     sfr_dict,
#     data_dict,
#     convolution_instruction,
#     sampled_data_dict,
#     position_sampling_function,
# ):
#     """
#     Function to handle position sampling function call

#     TODO: function calls like this can be generalized
#     TODO: have the function just update the sampled_data_dict instead
#     """

#     if position_sampling_function is not None:

#         # Construct what parameters are available for the extra function
#         available_parameters = {
#             "config": config,
#             "job_dict": job_dict,
#             "sfr_dict": sfr_dict,
#             "data_dict": data_dict,
#             "sampled_data_dict": sampled_data_dict,
#             "time_value": job_dict["convolution_time_bin_center"],
#             "convolution_instruction": convolution_instruction,
#             **convolution_instruction.get(
#                 "position_sampling_function_extra_parameters", {}
#             ),
#         }

#         # Make sure we extract the correct things from the available parameters
#         position_sampling_function_args = extract_arguments(
#             func=position_sampling_function,
#             arg_dict=available_parameters,
#         )

#         #
#         config["logger"].debug(
#             "Calculating positions using function {} and arguments {}".format(
#                 convolution_instruction["position_sampling_function"].__name__,
#                 position_sampling_function_args,
#             )
#         )

#         # Call position function
#         positions = position_sampling_function(**position_sampling_function_args)
#         if positions is None:
#             raise ValueError(
#                 "The position sampling function did not return a correct set of positions"
#             )

#         # add to dict
#         sampled_data_dict["positions"] = positions

#     return sampled_data_dict


def handle_post_convolution_function(
    config,
    job_dict,
    sfr_dict,
    data_dict,
    convolution_instruction,
    result_dict,
    name,
):
    """
    Function to handle post-convolution function call.

    An example of a post-convolution call is integrating systems to
    present-day time with LegWork and filtering out systems that do
    not fall within the LISA frequency range or that have merged by
    the present-day.

    Another example is to integrate systems through a gravitational
    potential based on the sampled position and a certain integration
    time.

    TODO: post_convolution_function does not have to be an
    argument. Can be extracted from the convolution_instruction itself
    """

    post_convolution_function = convolution_instruction.get(
        "post_convolution_function", None
    )

    if post_convolution_function is not None:

        # Construct what parameters are available for the extra function
        available_parameters = {
            "config": config,
            "job_dict": job_dict,
            "sfr_dict": sfr_dict,
            "data_dict": data_dict,
            "result_dict": result_dict,
            "time_value": job_dict["convolution_time_bin_center"],
            "convolution_instruction": convolution_instruction,
            **convolution_instruction.get(
                "post_convolution_function_extra_parameters", {}
            ),
        }

        # Make sure we extract the correct things from the available parameters
        post_convolution_function_args = extract_arguments(
            func=post_convolution_function,
            arg_dict=available_parameters,
        )

        #
        config["logger"].debug(
            "Handling '{}' post-convolution function call using function {} and arguments {}".format(
                name,
                convolution_instruction["post_convolution_function"].__name__,
                post_convolution_function_args,
            )
        )

        # Call function
        result_dict = post_convolution_function(**post_convolution_function_args)

        # check if the result dict is still a dict object
        if not isinstance(result_dict, dict):
            raise ValueError(
                "The result dict object must be a dictionary type object after the post-convolution call. It's now a {}-type object".format(
                    type(result_dict)
                )
            )

    return result_dict
