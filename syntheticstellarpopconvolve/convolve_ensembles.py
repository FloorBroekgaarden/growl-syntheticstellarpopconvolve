"""
Ensemble convolution functions

TODO: put all the general ensemble files into the ensemble file
TODO: allow the yield-rate and extra weights function to pass units to the ensemble and when
stripping the end-points we should make sure to handle that properly.
"""

import collections
import copy
import json
from collections import OrderedDict

import astropy.units as u
import h5py
import numpy as np

from syntheticstellarpopconvolve.convolution_default_settings import (
    ALLOWED_NUMERICAL_TYPES,
)
from syntheticstellarpopconvolve.convolution_general_functions import (
    calculate_digitized_sfr_rates,
    calculate_edge_values,
    handle_custom_scaling_or_conversion,
    handle_extra_weights_function,
)


class AutoVivificationDict(dict):
    """
    Implementation of perl's autovivification feature, by overriding the
    get item and the __iadd__ operator (https://docs.python.org/3/reference/datamodel.html?highlight=iadd#object.__iadd__)

    This allows to set values within a subdict that might not exist yet:

    Example:
        newdict = {}
        newdict['example']['mass'] += 10
        print(newdict)
        >>> {'example': {'mass': 10}}
    """

    def __getitem__(self, item):
        """
        Getitem function for the autovivication dict
        """

        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value

    def __iadd__(self, other):
        """
        iadd function (handling the +=) for the autovivication dict.
        """

        # if a value does not exist, assume it is 0.0
        try:
            self += other
        except:
            self = other
        return self


def merge_dicts(
    dict_1: dict,
    dict_2: dict,
    use_ordereddict=True,
    allow_matching_key_type_mismatch=True,
) -> dict:
    """
    Function to merge two dictionaries in a custom way. Taken from binarycpython

    Behaviour:

    When dict keys are only present in one of either:
        - we just add the content to the new dict

    When dict keys are present in both, we decide based on the value types how to combine them:
        - dictionaries will be merged by calling recursively calling this function again
        - numbers will be added
        - (opt) lists will be appended
        - booleans are merged with logical OR
        - identical strings are just set to the string
        - non-identical strings are concatenated
        - NoneTypes are set to None
        - In the case that the instances do not match: for now I will raise an error

    Args:
        dict_1: first dictionary
        dict_2: second dictionary

    Returns:
        Merged dictionary

    """

    # Set up new dict
    if use_ordereddict:
        new_dict = collections.OrderedDict()
    else:
        new_dict = {}

    ##################
    #
    keys_1 = dict_1.keys()
    keys_2 = dict_2.keys()

    ##################
    # Find overlapping keys of both dicts
    overlapping_keys = set(keys_1).intersection(set(keys_2))

    # Find the keys that are unique
    unique_to_dict_1 = set(keys_1).difference(set(keys_2))
    unique_to_dict_2 = set(keys_2).difference(set(keys_1))

    ##################
    # Add the unique keys to the new dict
    for key in unique_to_dict_1:
        # If these items are numerical or string, then just put them in
        if isinstance(dict_1[key], ALLOWED_NUMERICAL_TYPES + (str,)):
            new_dict[key] = dict_1[key]
        # Else, to be safe we should deepcopy them
        else:
            copy_dict = dict_1[key]
            new_dict[key] = copy_dict

    for key in unique_to_dict_2:
        # If these items are numerical or string, then just put them in
        if isinstance(dict_2[key], ALLOWED_NUMERICAL_TYPES + (str,)):
            new_dict[key] = dict_2[key]
        # Else, to be safe we should deepcopy them
        else:
            copy_dict = dict_2[key]
            new_dict[key] = copy_dict

    ##################
    # Go over the common keys:
    for key in overlapping_keys:

        ##################
        # If they keys are not the same, it depends on their type whether we still deal with them at all, or just raise an error
        if not isinstance(dict_1[key], type(dict_2[key])):

            ##################
            # Exceptions: numbers can be added
            if isinstance(dict_1[key], ALLOWED_NUMERICAL_TYPES) and isinstance(
                dict_2[key], ALLOWED_NUMERICAL_TYPES
            ):
                new_dict[key] = dict_1[key] + dict_2[key]

            ##################
            # Exceptions: versions of dicts can be merged
            elif isinstance(
                dict_1[key], (dict, collections.OrderedDict, type(AutoVivificationDict))
            ) and isinstance(
                dict_2[key], (dict, collections.OrderedDict, type(AutoVivificationDict))
            ):
                new_dict[key] = merge_dicts(
                    dict_1[key],
                    dict_2[key],
                    use_ordereddict=use_ordereddict,
                    allow_matching_key_type_mismatch=allow_matching_key_type_mismatch,
                )

            ##################
            #
            if not allow_matching_key_type_mismatch:
                print(
                    "Error key: {} value: {} type: {} and key: {} value: {} type: {} are not of the same type and cannot be merged".format(
                        key,
                        dict_1[key],
                        type(dict_1[key]),
                        key,
                        dict_2[key],
                        type(dict_2[key]),
                    )
                )
                raise ValueError

            ##################
            # one key is None, just use the other
            elif dict_1[key] is None:
                try:
                    new_dict[key] = dict_2[key]
                except:
                    msg = f"{key}: Failed to set from {dict_2[key]} when other key was of NoneType "
                    raise ValueError(msg)

            elif dict_1[key] is None:
                try:
                    new_dict[key] = dict_1[key]
                except:
                    msg = f"{key}: Failed to set from {dict_1[key]} when other key was of NoneType "
                    raise ValueError(msg)

            # string-int clash : convert both to ints and save
            elif (
                isinstance(dict_1[key], str)
                and isinstance(dict_2[key], int)
                or isinstance(dict_1[key], int)
                and isinstance(dict_2[key], str)
            ):
                try:
                    new_dict[key] = int(dict_1[key]) + int(dict_2[key])
                except ValueError as e:
                    msg = "{}: Failed to convert string (either '{}' or '{}') to an int".format(
                        key, dict_1[key], dict_2[key]
                    )
                    raise ValueError(msg) from e

            # string-float clash : convert both to floats and save
            elif (
                isinstance(dict_1[key], str)
                and isinstance(dict_2[key], float)
                or isinstance(dict_1[key], float)
                and isinstance(dict_2[key], str)
            ):
                try:
                    new_dict[key] = float(dict_1[key]) + float(dict_2[key])
                except ValueError as e:
                    msg = "{}: Failed to convert string (either '{}' or '{}') to an float".format(
                        key, dict_1[key], dict_2[key]
                    )
                    raise ValueError(msg) from e

            # If the above cases have not dealt with it, then we should raise an error
            else:
                msg = "merge_dicts error: key: {key} value: {value1} type: {type1} and key: {key} value: {value2} type: {type2} are not of the same type and cannot be merged".format(
                    key=key,
                    value1=dict_1[key],
                    type1=type(dict_1[key]),
                    value2=dict_2[key],
                    type2=type(dict_2[key]),
                )
                raise ValueError(msg)

        # Here the keys are the same type
        # Here we check for the cases that we want to explicitly catch. Ints will be added,
        # floats will be added, lists will be appended (though that might change) and dicts will be
        # dealt with by calling this function again.
        else:
            # ints
            # Booleans (has to be the type Bool, not just a 0 or 1)
            if isinstance(dict_1[key], bool) and isinstance(dict_2[key], bool):
                new_dict[key] = dict_1[key] or dict_2[key]

            elif isinstance(dict_1[key], int) and isinstance(dict_2[key], int):
                new_dict[key] = dict_1[key] + dict_2[key]

            elif isinstance(dict_1[key], np.int64) and isinstance(
                dict_2[key], np.int64
            ):
                new_dict[key] = dict_1[key] + dict_2[key]

            # floats
            elif isinstance(dict_1[key], float) and isinstance(dict_2[key], float):
                new_dict[key] = dict_1[key] + dict_2[key]

            # lists
            elif isinstance(dict_1[key], list) and isinstance(dict_2[key], list):
                new_dict[key] = dict_1[key] + dict_2[key]

            # Astropy quantities (using a dummy type representing the numpy array)
            elif isinstance(dict_1[key], type(np.array([1]) * u.m)) and isinstance(
                dict_2[key], type(np.array([1]) * u.m)
            ):
                new_dict[key] = dict_1[key] + dict_2[key]

            # dicts
            elif isinstance(dict_1[key], dict) and isinstance(dict_2[key], dict):
                new_dict[key] = merge_dicts(
                    dict_1[key],
                    dict_2[key],
                    use_ordereddict=use_ordereddict,
                    allow_matching_key_type_mismatch=allow_matching_key_type_mismatch,
                )

            # strings
            elif isinstance(dict_1[key], str) and isinstance(dict_2[key], str):
                if dict_1[key] == dict_2[key]:
                    # same strings
                    new_dict[key] = dict_1[key]
                else:
                    # different strings: just concatenate them
                    new_dict[key] = dict_1[key] + dict_2[key]

            # None types
            elif dict_1[key] is None and dict_2[key] is None:
                new_dict[key] = None

            else:
                msg = "Object types {}: {} ({}), {} ({}) not supported.".format(
                    key,
                    dict_1[key],
                    type(dict_1[key]),
                    dict_2[key],
                    type(dict_2[key]),
                )
                raise ValueError(msg)

    #
    return new_dict


def multiply_ensemble(ensemble, factor):
    """
    Function to multiply the endpoints for
    """

    for key in ensemble.keys():
        if isinstance(ensemble[key], dict):
            multiply_ensemble(ensemble=ensemble[key], factor=factor)
        elif isinstance(ensemble[key], (int, float)):
            ensemble[key] = ensemble[key] * factor


def check_if_value_layer(keys):
    """
    Function to determine whether the supplied set of keys consists of value-keys or name-keys
    """

    is_value_layer = True
    for key in keys:
        try:
            float(key)
        except ValueError:
            is_value_layer = False
    return is_value_layer


def get_layer_iterable(ensemble, is_value_layer):
    """
    Function to get the layer iterable
    """

    ########
    # determine iterable
    if is_value_layer:
        iterable = sorted(ensemble.keys(), key=lambda x: float(x))
    else:
        iterable = ensemble.keys()
    return iterable


def check_if_value_layer_and_get_layer_iterable(ensemble):
    """
    Function to handle checking whether the layer is a data layer and then build up the iterable
    """

    #
    is_value_layer = check_if_value_layer(ensemble.keys())
    iterable = get_layer_iterable(ensemble, is_value_layer)

    return is_value_layer, iterable


def handle_binsize_multiplication_factor(
    config,
    ensemble,
    data_layer_dict_entry,
    key,
    key_i,
    binsizes,
    name,
    extra_value_dict,
):
    """
    Function to handle the calculation of the binsize multiplication
    """

    ##########
    # check if we want to multiply this by the binsizes of the ensemble
    if data_layer_dict_entry.get("multiply_by_binsize", False):
        # Determine binsizes

        # Calculate the binsizes (otherwise assume its already calculated)
        if binsizes is None:
            binsizes = get_ensemble_binsizes(
                config=config,
                ensemble=ensemble,
                data_layer_dict_entry=data_layer_dict_entry,
            )

            #
            config["logger"].debug("Calculated binsizes: {}".format(binsizes))

        # select current binsize
        binsize = binsizes[key_i]

        #
        config["logger"].debug(
            "Selected binsize {} for {} ({})".format(binsize, key, key_i)
        )

        # # TODO: check if the binsize extends beyond the last starformation
        # binsize = restrict_binsize()
        # config["logger"].debug("Restricted binsize to {}".format(binsize))

        #
        extra_value_dict["{}_binsize".format(name)] = binsize

    return binsizes, extra_value_dict


def get_ensemble_binsizes(config, ensemble, data_layer_dict_entry):
    """
    Function to calculate the binsizes, taking into account transformations of the numbers.
    """

    #######
    # TODO: make this optional
    calculate_edges_before_transformations = True

    #######
    # if binsizes are provided then just use those
    if "binsizes" in data_layer_dict_entry:
        return data_layer_dict_entry["binsizes"]

    #######
    # loop over sorted (float) version of the ensemble keys:
    sorted_ensemble_keys = list(sorted(ensemble.keys(), key=lambda x: float(x)))

    #
    values = np.array([float(el) for el in sorted_ensemble_keys])

    # Transform edges
    if calculate_edges_before_transformations:

        # calculate edge values
        edge_values = calculate_edge_values(values)

        # perform transformations
        transformed_edge_values = np.array(
            [
                handle_custom_scaling_or_conversion(
                    config=config,
                    data_layer_or_column_dict_entry=data_layer_dict_entry,
                    value=edge_value,
                )
                for edge_value in edge_values
            ]
        )
    else:
        # perform transformations
        transformed_values = np.array(
            [
                handle_custom_scaling_or_conversion(
                    config=config,
                    data_layer_or_column_dict_entry=data_layer_dict_entry,
                    value=value,
                )
                for value in values
            ]
        )

        # calculate edge values
        transformed_edge_values = calculate_edge_values(transformed_values)

    # calculate binsizes
    binsizes = np.diff(transformed_edge_values)

    #
    config["logger"].debug(
        "Values: {} transformed_edge_values: {} binsizes: {}".format(
            values, transformed_edge_values, binsizes
        )
    )

    return binsizes


################
# data-layer dict functionality
def get_data_layer_dict_values(data_layer_dict):
    """
    function to extract data layer values
    """

    data_layer_values = []

    for key in data_layer_dict:
        if isinstance(data_layer_dict[key], int):
            data_layer_values.append(data_layer_dict[key])
        elif isinstance(data_layer_dict[key], dict):
            data_layer_values.append(data_layer_dict[key]["layer_depth"])
        else:
            raise ValueError("input type not supported.")

    return data_layer_values


def get_deepest_data_layer_depth(data_layer_dict):
    """
    Function to get the deepest data layer depth
    """

    data_layer_values = get_data_layer_dict_values(data_layer_dict=data_layer_dict)

    return max(data_layer_values)


def shift_layers_dict(data_layer_dict, shift_value):
    """
    Function to shift the layer depths
    """

    new_layer_depth_dict = copy.copy(data_layer_dict)

    for key in data_layer_dict.keys():
        shift_data_layer(
            data_layer_dict=new_layer_depth_dict, key=key, shift=shift_value
        )

    return new_layer_depth_dict


def shift_data_layer(data_layer_dict, key, shift):
    """
    Function to shift the data layer with a certain value
    """

    if isinstance(data_layer_dict[key], int):
        data_layer_dict[key] += shift
    elif isinstance(data_layer_dict[key], dict):
        data_layer_dict[key]["layer_depth"] += shift
    else:
        raise ValueError("input type not supported.")


def invert_data_layer_dict(data_layer_dict):
    """
    Function to invert the data layer dictionary.

    This function does not truly swap key and values because if entries in the
    data_layer_dict contain dictionaries we will extract the layer depth from
    them instead of making the dictionary the key.
    """

    inverted_data_layer_dict = {}

    for key in data_layer_dict:
        if isinstance(data_layer_dict[key], int):
            inverted_data_layer_dict[data_layer_dict[key]] = key
        elif isinstance(data_layer_dict[key], dict):
            inverted_data_layer_dict[data_layer_dict[key]["layer_depth"]] = key
        else:
            raise ValueError("input type not supported.")

    return inverted_data_layer_dict


##################
# other
def shift_layers_list(layer_list, shift_value):
    """
    Function to shift the entries in a list by a certain value
    """

    for el_i, _ in enumerate(layer_list):
        layer_list[el_i] += shift_value

    return layer_list


################
# endpoints functionality
def extract_endpoints(ensemble, endpoint_list=None):
    """
    Function to strip the endpoints
    """

    if endpoint_list is None:
        endpoint_list = []

    # Handle recursive
    if isinstance(ensemble, (dict, OrderedDict)):
        for key in ensemble.keys():
            if isinstance(ensemble[key], (dict, OrderedDict)):
                endpoint_list = extract_endpoints(
                    ensemble[key], endpoint_list=list(endpoint_list)
                )
            elif isinstance(ensemble[key], (int, float)):
                endpoint_list.append(ensemble[key])
        return endpoint_list

    return np.array(endpoint_list)


def attach_endpoints(ensemble, endpoint_array, counter=0, depth=0):
    """
    Function to attach endpoints
    """

    #
    if isinstance(ensemble, (dict, OrderedDict)):

        #
        for key in ensemble.keys():
            if isinstance(ensemble[key], (dict, OrderedDict)):
                counter = attach_endpoints(
                    ensemble[key],
                    endpoint_array=endpoint_array,
                    counter=counter,
                    depth=depth + 1,
                )
            if ensemble[key] is None:
                ensemble[key] = endpoint_array[counter]
                counter += 1

        # if we're back all the way up and we've not exhausted the array then something is wrong
        if depth == 0:
            if len(endpoint_array) != counter:
                raise ValueError(
                    "Somehow we have not reached the end of the endpoint array"
                )
        return counter

    raise ValueError("Unsupported type: {}".format(type(ensemble)))


def set_endpoints(ensemble, value):
    """
    Function to set the endpoints to a fixed value
    """

    if isinstance(ensemble, (dict, OrderedDict)):
        for key in ensemble.keys():
            if isinstance(ensemble[key], (dict, OrderedDict)):
                set_endpoints(ensemble=ensemble[key], value=value)
            elif isinstance(ensemble[key], (int, float)):
                ensemble[key] = value
            elif ensemble[key] is None:
                ensemble[key] = value

        if len(ensemble.keys()) == 0:
            raise ValueError("Encountered empty ensemble")

    else:
        raise ValueError("Unsupported type: {}".format(type(ensemble)))


def strip_ensemble_endpoints(ensemble):
    """
    Function to strip the ensemble from its endpoints
    """

    # extract endpoints
    endpoints = extract_endpoints(ensemble=ensemble)

    # set original ensemble endpoints to 0
    set_endpoints(ensemble=ensemble, value=0)

    return ensemble, endpoints


################
def get_depth_ensemble_first_endpoint(ensemble, depth=0):
    """
    Function to get the maximum depth in an ensemble. Stops at the first branch
    that does not contain a key or nested dict (i.e. this function does not crawl through the entire ensemble)
    """

    if isinstance(ensemble, (dict)):
        for key in ensemble.keys():
            return get_depth_ensemble_first_endpoint(ensemble[key], depth=depth + 1)
    else:
        return depth


def get_max_depth_ensemble(ensemble):
    """
    Function to find the maximum depth in a nested dictionary
    """

    def _find_max_depth_helper(ensemble, depth):  # DH0001
        if not isinstance(ensemble, dict):
            return depth
        max_depth = depth
        for value in ensemble.values():
            max_depth = max(max_depth, _find_max_depth_helper(value, depth + 1))
        return max_depth

    return _find_max_depth_helper(ensemble, 0)


def get_depth_ensemble_all_endpoints(ensemble, depth=0, endpoint_depths=None):
    """
    Function to get the maximum depth in an ensemble. Stops at the first branch
    that does not contain a key or nested dict (i.e. this function does not crawl through the entire ensemble)
    """

    if endpoint_depths is None:
        endpoint_depths = []

    if isinstance(ensemble, (dict)):
        for key in ensemble.keys():
            get_depth_ensemble_all_endpoints(
                ensemble[key], depth=depth + 1, endpoint_depths=endpoint_depths
            )
    else:
        endpoint_depths.append(depth)
    return endpoint_depths


################
# Functions to get the ensemble structure.
def _get_ensemble_structure(ensemble, structure_dict, max_depth, depth=0):
    """
    Recursive function acompanying the "get_ensemble_structure" function
    """

    #
    if depth < max_depth:
        for key in ensemble.keys():
            # add to structure
            if key not in structure_dict[depth]:
                structure_dict[depth].append(key)

            # check the rest
            structure_dict = _get_ensemble_structure(
                ensemble=ensemble[key],
                structure_dict=structure_dict,
                max_depth=max_depth,
                depth=depth + 1,
            )

    return structure_dict


def get_ensemble_structure(ensemble, named_layer_list=None):
    """
    Function to generate the ensemble structure.

    Optionally, if a named layer list is provided, this function will raise an error if named layers contain more than one value
    """

    # check if there are endpoints of varying depth
    depth_all_endpoints = get_depth_ensemble_all_endpoints(ensemble)

    if len(np.unique(depth_all_endpoints)) > 1:
        raise ValueError("This ensemble has endpoints of varying depth. abort")

    #
    max_depth = depth_all_endpoints[0]

    # setup structure dict
    structure_dict = {el: [] for el in range(max_depth)}

    # generate structure
    structure_dict = _get_ensemble_structure(
        ensemble=ensemble, structure_dict=structure_dict, max_depth=max_depth
    )

    # check content if necessary
    if named_layer_list is not None:
        for named_layer_depth in named_layer_list:
            if len(structure_dict[named_layer_depth]) > 1:
                raise ValueError("Multiple values at named layer depth")

    return structure_dict


##############
# marginaliser functions


def ensemble_marginalise_layer(ensemble, marginalisation_depth, depth=0):
    """
    Function to marginalise a layer of the ensemble
    """

    if marginalisation_depth == 0:
        raise ValueError("We can't currently marginalise with marginalisation_depth=-1")

    if depth + 1 == marginalisation_depth:
        # merge subdicts
        for key in ensemble.keys():
            merged_subdicts = {}
            for subdict_key in ensemble[key].keys():
                merged_subdicts = merge_dicts(
                    merged_subdicts,
                    ensemble[key][subdict_key],
                    use_ordereddict=False,
                    allow_matching_key_type_mismatch=False,
                )
            ensemble[key] = merged_subdicts
    else:
        for key in ensemble.keys():
            ensemble[key] = ensemble_marginalise_layer(
                ensemble=ensemble[key],
                depth=depth + 1,
                marginalisation_depth=marginalisation_depth,
            )

    return ensemble


def ensemble_handle_marginalisation(
    config, ensemble, convolution_instruction, is_pre_conv
):
    """
    Function to handle ensemble marginalisation pre and post convolution.

    Inspects the marginalisation layer dict and marginalises deepest-first

    If a data dict is provided and the 'is_pre_conv' flag is set to true, layers that match those in the data dict get skipped
    """

    ########
    # if there is no marginalisation provided, we don't really have to do anything
    if "marginalisation_list" not in convolution_instruction.keys():
        return config, ensemble, convolution_instruction

    #
    config["logger"].debug(
        "Handling ensemble marginalisation (is_pre_conv: {})".format(is_pre_conv)
    )

    ########
    # find out which layers we can actually remove (to avoid removing the actual data we need to convolve.)
    to_remove_layers = []
    for layer in convolution_instruction["marginalisation_list"]:
        # skip if the current layer exists in the data layer dict and we are doing pre-convolution
        if is_pre_conv:
            if layer in get_data_layer_dict_values(
                data_layer_dict=convolution_instruction["data_layer_dict"]
            ):
                continue
        to_remove_layers.append(layer)

    #########
    # Store which layers we should keep
    to_keep_layers = list(
        set(convolution_instruction["marginalisation_list"]) - set(to_remove_layers)
    )
    convolution_instruction["marginalisation_list"] = to_keep_layers

    # sort list so we do deepest first
    to_remove_layers = sorted(to_remove_layers, reverse=True)

    #########
    # Process layers
    for layer in to_remove_layers:
        #
        config["logger"].debug("Marginalising layer {}".format(layer))

        # perform marginalisation
        ensemble = ensemble_marginalise_layer(
            ensemble=ensemble, marginalisation_depth=layer
        )

        # update any data layer dict entry with an updated depth
        if is_pre_conv:
            for key, value in convolution_instruction["data_layer_dict"].items():
                if value > layer:
                    shift_data_layer(
                        data_layer_dict=convolution_instruction["data_layer_dict"],
                        key=key,
                        shift=-1,
                    )

        # Also update the part of the marginalisation list that will not currently be handled
        for index, value in enumerate(convolution_instruction["marginalisation_list"]):
            if value > layer:
                convolution_instruction["marginalisation_list"][index] = value - 1

    return config, ensemble, convolution_instruction


###########
# Main ensemble convolution functions
def extract_ensemble_data(config, convolution_instruction):
    """
    Function to extract the ensemble-type data
    """

    #
    config["logger"].debug(
        "Extracting ensemble data {}".format(convolution_instruction["input_data_name"])
    )

    data_dict = {}
    with h5py.File(config["output_filename"], "r") as output_hdf5file:
        data_dict["ensemble_data"] = {
            convolution_instruction["input_data_name"]: json.loads(
                output_hdf5file[
                    "input_data/ensemble/{}".format(
                        convolution_instruction["input_data_name"]
                    )
                ][()]
            )
        }

    ##########
    # we need to shift the layers because the way the data gets stored is by
    # adding a layer at the start
    convolution_instruction["data_layer_dict"] = shift_layers_dict(
        convolution_instruction["data_layer_dict"], shift_value=1
    )
    if "marginalisation_list" in convolution_instruction:
        convolution_instruction["marginalisation_list"] = shift_layers_list(
            convolution_instruction["marginalisation_list"], shift_value=1
        )

    ##########
    # handle pre-convolution ensemble marginalisation
    (
        config,
        data_dict["ensemble_data"],
        convolution_instruction,
    ) = ensemble_handle_marginalisation(
        config=config,
        ensemble=data_dict["ensemble_data"],
        convolution_instruction=convolution_instruction,
        is_pre_conv=True,
    )

    return config, data_dict, convolution_instruction


def ensemble_handle_SFR_multiplication(
    convolution_time_bin_center,
    job_dict,
    config,
    convolution_instruction,
    ensemble,
    data_dict,
    extra_value_dict=None,
):
    """
    Function to handle multiplying the provided ensemble with a.
    """

    #
    if extra_value_dict is None:
        extra_value_dict = {}
    extra_value = np.prod(list(extra_value_dict.values()))

    #
    config["logger"].debug(
        "Convolving ensemble data with SFR rate at convolution bin {} with data_dict: {} and multiplying by {} ({})".format(
            convolution_time_bin_center, data_dict, extra_value, extra_value_dict
        )
    )

    ##############
    # to re-use the event-based functionality we should cast all the data in the data_dict into numpy arrays
    new_data_dict = {}
    for key, value in data_dict.items():
        try:
            new_data_dict[key] = np.array([value])
        except TypeError:
            new_data_dict[key] = np.array([value.value]) * value.unit
    data_dict = new_data_dict

    #############
    digitized_sfr_rates = calculate_digitized_sfr_rates(
        config=config,
        convolution_time_bin_center=convolution_time_bin_center,
        data_dict=data_dict,
        sfr_dict=job_dict["sfr_dict"],
    ).value

    ################
    # Run custom function afterwards
    extra_weights = handle_extra_weights_function(
        config=config,
        convolution_time_bin_center=convolution_time_bin_center,
        convolution_instruction=convolution_instruction,
        data_dict=data_dict,
        output_shape=np.array([1]).shape,
    )

    # Combine SFR, extra weight and possibly time-duration (time bin-width) to turn rates into numbers
    combined = digitized_sfr_rates[0] * extra_weights[0] * extra_value

    # Multiply ensemble with that number
    multiply_ensemble(ensemble=ensemble, factor=combined)

    return ensemble


def ensemble_convolve_ensemble(
    convolution_time_bin_center,
    job_dict,
    config,
    convolution_instruction,
    ensemble,
    depth=0,
    data_dict=None,
    extra_value_dict=None,
):
    """
    Recursive function that handles convolving the ensemble.
    """

    ########
    #
    if data_dict is None:
        data_dict = {}
    if extra_value_dict is None:
        extra_value_dict = {}
    binsizes = None

    ########
    # unpack some data
    data_layer_dict = convolution_instruction["data_layer_dict"]
    inverted_data_layer_dict = convolution_instruction["inverted_data_layer_dict"]
    deepest_data_layer_depth = convolution_instruction["deepest_data_layer_depth"]
    data_layer_values = convolution_instruction["data_layer_values"]

    #
    config["logger"].debug(
        "Convolving ensemble at depth {} using data_dict: {} data_layer_dict: {}, deepest_data_layer_depth: {}, data_layer_values: {}".format(
            depth,
            data_dict,
            data_layer_dict,
            deepest_data_layer_depth,
            data_layer_values,
        )
    )

    ########
    #
    if isinstance(ensemble, dict):

        ########
        # check if we are in a value-layer
        is_value_layer, layer_iterable = check_if_value_layer_and_get_layer_iterable(
            ensemble
        )
        config["logger"].debug("Iterable: {}".format(layer_iterable))

        ########
        # Go over the keys
        for key_i, key in enumerate(layer_iterable):
            config["logger"].debug("Current layer key: {} ({})".format(key, key_i))

            #################
            # check if the layer is one of the target depths to pick up data for the data dict and to find
            if depth in data_layer_values:
                #
                name = inverted_data_layer_dict[depth]

                config["logger"].debug(
                    "The layer ({}) is that of data layer {}. Storing value {} in data_dict under {}".format(
                        depth,
                        name,
                        key,
                        name,
                    )
                )

                ###########
                # if we are in a data layer but also this layer is not a value layer then there is an issue
                if not is_value_layer:
                    raise ValueError(
                        "The current layer has not been picked up as a value layer, but is configured to be a data-layer. Please check your data-layer dict."
                    )

                ###########
                # if its a integer we just assume the current layer we hit does not have to be converted (other than to float)
                if isinstance(data_layer_dict[name], int):
                    value = float(key)

                    #################
                    # Handle unit for delay-time
                    if name == "delay_time":
                        value = value * config["delay_time_default_unit"]

                # if its a dictionary, we have more options: convert with factor, convert with function, calculate binsize
                elif isinstance(data_layer_dict[name], dict):

                    ########
                    #
                    value = float(key)

                    ########
                    # multiply by factor or apply function on value
                    value = handle_custom_scaling_or_conversion(
                        config=config,
                        data_layer_or_column_dict_entry=data_layer_dict[name],
                        value=value,
                    )

                    #################
                    # Handle unit for delay-time
                    if name == "delay_time":
                        if "delay_time_unit" in data_layer_dict[name].keys():
                            value = value * data_layer_dict[name]["delay_time_unit"]
                        else:
                            value = value * config["delay_time_default_unit"]

                    ########
                    # Determine binsize multiplication factor
                    binsizes, extra_value_dict = handle_binsize_multiplication_factor(
                        config=config,
                        ensemble=ensemble,
                        data_layer_dict_entry=data_layer_dict[name],
                        name=name,
                        key=key,
                        key_i=key_i,
                        binsizes=binsizes,
                        extra_value_dict=extra_value_dict,
                    )
                #
                else:
                    raise ValueError("input type not supported.")

                # store
                data_dict[inverted_data_layer_dict[depth]] = value

                #
                config["logger"].debug("data dict: {}".format(data_dict))

            #################
            # multiply with starformation if we reached a depth that is deeper than any data layer.
            if depth >= deepest_data_layer_depth:
                # multiplication with SFR-related things here
                ensemble[key] = ensemble_handle_SFR_multiplication(
                    convolution_time_bin_center=convolution_time_bin_center,
                    job_dict=job_dict,
                    config=config,
                    convolution_instruction=convolution_instruction,
                    ensemble=ensemble[key],
                    data_dict=data_dict,
                    extra_value_dict=extra_value_dict,
                )
            else:
                # call self with increased depth
                ensemble[key] = ensemble_convolve_ensemble(
                    convolution_time_bin_center=convolution_time_bin_center,
                    job_dict=job_dict,
                    config=config,
                    convolution_instruction=convolution_instruction,
                    ensemble=ensemble[key],
                    depth=depth + 1,
                    data_dict=data_dict,
                    extra_value_dict=extra_value_dict,
                )

    elif isinstance(ensemble, ALLOWED_NUMERICAL_TYPES):
        raise ValueError(
            "Arrived at a layer in the ensemble that is of numerical type (depth={}), likely the endpoint. This should not happen".format(
                depth
            )
        )

    return ensemble


def ensemble_convolution_function(
    convolution_time_bin_center, job_dict, config, convolution_instruction, data_dict
):
    """
    Function for the multiprocessing worker to convolve ensemble-based data.

    There are some requirements for the ensemble structure, but the nested
    dictionary that is passed by data_dict needs to contain at least time or
    log10_time

    Moreover, the end-point nodes are expected to contain the
    quantity-per-unit-mass. In that way we do not have to rely on extracting
    that from the meta-data and stuff
    """

    #
    config["logger"].debug(
        "Convolving ensemble-based data {} for bin_center {}".format(
            convolution_instruction["input_data_name"], convolution_time_bin_center
        )
    )

    # pre-convolution preparation
    ensemble = data_dict["ensemble_data"]
    data_layer_dict = convolution_instruction["data_layer_dict"]

    # check if we want to supply a fixed metallicity
    data_dict = {}
    if "metallicity_value" in convolution_instruction:
        data_dict["metallicity"] = convolution_instruction["metallicity_value"]

    # add some extra things to the convolution instruction TODO this can be placed elsewhere? TODO: what the difference between max_depth and deepest_data_layer_depth?
    convolution_instruction["deepest_data_layer_depth"] = get_deepest_data_layer_depth(
        data_layer_dict=data_layer_dict
    )
    convolution_instruction["inverted_data_layer_dict"] = invert_data_layer_dict(
        data_layer_dict=data_layer_dict
    )
    convolution_instruction["data_layer_values"] = get_data_layer_dict_values(
        data_layer_dict=data_layer_dict
    )

    # convolution
    ensemble = ensemble_convolve_ensemble(
        ensemble=ensemble,
        convolution_instruction=convolution_instruction,
        config=config,
        convolution_time_bin_center=convolution_time_bin_center,
        job_dict=job_dict,
        data_dict=data_dict,
    )

    # marginalisation
    config, ensemble, convolution_instruction = ensemble_handle_marginalisation(
        config=config,
        ensemble=ensemble,
        convolution_instruction=convolution_instruction,
        is_pre_conv=False,
    )

    # detach endpoints from ensemble
    stripped_ensemble, stripped_endpoints = strip_ensemble_endpoints(ensemble=ensemble)

    # return endpoints and ensemble if first job
    result_dict = {"convolution_result": stripped_endpoints}
    if job_dict["job_number"] == 0:
        result_dict["stripped_ensemble"] = stripped_ensemble

    return result_dict
