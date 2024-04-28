def extract_custom_data(config, convolution_instruction):
    """
    Function that handles extracting the custom data. Requires the user to provide a hook 'custom_data_extraction_function' in the config.
    """

    raise NotImplementedError

    return config["custom_data_extraction_function"](
        config=config, convolution_instruction=convolution_instruction
    )


def custom_convolution_function(
    time_value, job_dict, config, convolution_instruction, data_dict
):
    """
    Custom convolution function. Requries the user to provide a hook 'custom_convolution_function' in the config.
    """

    raise NotImplementedError

    return config["custom_convolution_function"](
        time_value=time_value,
        job_dict=job_dict,
        config=config,
        convolution_instruction=convolution_instruction,
        data_dict=data_dict,
    )
