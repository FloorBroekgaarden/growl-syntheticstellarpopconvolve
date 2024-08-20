"""
File containing the default convolution-instruction.
"""

import copy
import logging
import os
from typing import Callable

import astropy.units as u
import numpy as np
import voluptuous as vol
from astropy.cosmology import Planck13 as cosmo  # Planck 2013

ALLOWED_NUMERICAL_TYPES = (int, float, complex, np.number)
dimensionless_unit = u.m / u.m

default_convolution_instruction_dict = {""}


# extract only values
default_convolution_instruction = {
    key: value["value"] for key, value in default_convolution_instruction_dict.items()
}

# extract only descriptions
default_convolution_instruction_descriptions = {
    key: value["description"]
    for key, value in default_convolution_instruction_dict.items()
}

#############
# Utilities to build the description table


# TODO: quite a bit of overlap here with the other code. maybe store somewhere else.
def build_description_table(table_name, parameter_list, description_dict):
    """
    Function to create a table containing the description of the options
    """

    #
    indent = "   "

    # Get parameter list and parse descriptions
    parameter_list_with_descriptions = [
        [
            parameter,
            parse_description(description_dict=description_dict[parameter]),
        ]
        for parameter in parameter_list
    ]

    # Construct parameter list
    rst_table = """
.. list-table:: {}
{}:widths: 25, 75
{}:header-rows: 1
""".format(
        table_name, indent, indent
    )

    #
    rst_table += "\n"
    rst_table += indent + "* - Option\n"
    rst_table += indent + "  - Description\n"

    for parameter_el in parameter_list_with_descriptions:
        rst_table += indent + "* - {}\n".format(parameter_el[0])
        rst_table += indent + "  - {}\n".format(parameter_el[1])

    return rst_table


def parse_description(description_dict):
    """
    Function to parse the description for a given parameter
    """

    # Make a local copy
    description_dict = copy.copy(description_dict)

    ############
    # Add description
    description_string = "Description:\n   "

    # Clean description text
    description_text = description_dict["description"].strip()

    if description_text:
        description_text = description_text[0].capitalize() + description_text[1:]
        if description_text[-1] != ".":
            description_text = description_text + "."
    description_string += description_text

    ##############
    # Add unit (in latex)
    if "unit" in description_dict:
        if description_dict["unit"] != dimensionless_unit:
            description_string = description_string + "\n\nUnit: [{}].".format(
                description_dict["unit"].to_string("latex_inline")
            )

    ##############
    # Add default value
    if "value" in description_dict:
        # Clean
        if isinstance(description_dict["value"], str) and (
            "/home" in description_dict["value"]
        ):
            description_dict["value"] = "example path"

        # Write
        description_string = description_string + "\n\nDefault value:\n   {}".format(
            description_dict["value"]
        )

    ##############
    # Add validation
    if "validation" in description_dict:
        # Write
        description_string = description_string + "\n\nValidation:\n   {}".format(
            description_dict["validation"]
        )

    # Check if there are newlines, and replace them by newlines with indent
    description_string = description_string.replace("\n", "\n       ")

    return description_string


def write_default_settings_to_rst_file(options_defaults_dict, output_file: str) -> None:
    """
    Function that writes the descriptions of the grid options to an rst file

    Args:
        output_file: target file where the grid options descriptions are written to
    """

    ###############
    # Check input
    if not output_file.endswith(".rst"):
        msg = "Filename doesn't end with .rst, please provide a proper filename"
        raise ValueError(msg)

    ###############
    # construct descriptions dict
    descriptions_dict = {}
    for key, value in options_defaults_dict.items():
        descriptions_dict[key] = {}
        descriptions_dict[key]["description"] = value["description"]
        descriptions_dict[key]["value"] = value["value"]

        if "validation" in value:
            descriptions_dict[key]["validation"] = value["validation"]

    # separate public and private options
    public_options = [key for key in descriptions_dict if not key.startswith("_")]
    # private_options = [key for key in descriptions_dict if key.startswith("_")]

    ###############
    # Build description page text

    # Set up intro
    description_page_text = ""
    title = "Convolution options"
    description_page_text += title + "\n"
    description_page_text += "=" * len(title) + "\n\n"
    description_page_text += "The following chapter contains all Population code options, along with their descriptions."
    description_page_text += "\n\n"

    # Set up description table for the public options
    public_options_description_title = "Public options"
    public_options_description_text = public_options_description_title + "\n"
    public_options_description_text += (
        "-" * len(public_options_description_title) + "\n\n"
    )
    public_options_description_text += "In this section we list the public options for the population code. These are meant to be changed by the user.\n"
    public_options_description_text += build_description_table(
        table_name="Public options",
        parameter_list=sorted(public_options),
        description_dict=descriptions_dict,
    )
    description_page_text += public_options_description_text
    description_page_text += "\n\n"

    # # Set up description table for the private options
    # private_options_description_title = "Private internal variables"
    # private_options_description_text = private_options_description_title + "\n"
    # private_options_description_text += (
    #     "-" * len(private_options_description_title) + "\n\n"
    # )
    # private_options_description_text += "In this section we list the private internal parameters for the population code. These are not meant to be changed by the user.\n"
    # private_options_description_text += build_description_table(
    #     table_name="Private internal variables",
    #     parameter_list=sorted(private_options),
    #     description_dict=descriptions_dict,
    # )
    # description_page_text += private_options_description_text
    # description_page_text += "\n\n"

    ###############
    # write to file
    with open(output_file, "w") as f:
        f.write(description_page_text)
