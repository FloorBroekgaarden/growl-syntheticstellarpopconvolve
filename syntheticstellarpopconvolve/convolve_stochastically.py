"""Routines for stochastic convolution


initial idea with simple situation

sfr [Msun /yr], global
fixed Z

partially grid-like

with a SFR evaluated in lookback time in bins, with t_l,i lookback times and dt_l,i binsize and edges t_l,i-0.5, t_l,i+0.5

in a given bin we have the total mass formed in stars sfr(t_l=t_l,i) * dt_l,i = m_tot,i

now, we have some systems of interest (e.g. dwd), gained through pop-synth simulations. these have a.o. the property normalized yield, i..e number per formed solar mass

Y_j [Msun]

total number of system j sampled;
Y_j * M_tot,i = N_j

if N_j > 1:
- take X systems where X = floor(N_j)
- N_j-x is then < 1
- take random number from uniform dist, P. if P < N_j-x: accept, else not

then we have a bunch of systems (which can include the same system)
but in that array, assign random lookback time between the bin edges

assign radnom position

- this sampling stategy can be multiprocssed easily (lookback time bins)
- can also easily be extended to include metallicity
- naturally handles unequal yield per systems



Notes:
- this method does not turn things around like the others do. We start
at a given lookback time bin for all systems. We sample a set of
systems based on the total starformation within that lookback time
bin, and the normalized yields of the systems. We then assign a birth
lookback time to the systems (taken randomly between the bin edges)

TODO: consider putting the configuration in through the convolution instruction rather than the global config.
"""

import time
import uuid

import astropy.units as u
import numpy as np

from syntheticstellarpopconvolve.general_functions import extract_arguments


def combine_dicts_with_numpy_array_entries(dict1, dict2):
    """
    Function to combine dicts that contain numpy-array entries. Loops over the keys in dict2 and stores in or appends to the same entry in dict1

    TODO: merge with convolve_ensembles.merge_dicts
    """

    # print("dict1, dict2", dict1, dict2)
    # print("pre: len dict1, len dict2", len(dict1.get('IDs', [])), len(dict2.get('IDs', [])))
    for key in dict2.keys():
        if key not in dict1.keys():
            dict1[key] = dict2[key]
        else:
            dict1[key] = np.concatenate([dict1[key], dict2[key]])

    # print("dict1, dict2", dict1, dict2)
    # print("post: len dict1, len dict2", len(dict1.get('IDs', [])), len(dict2.get('IDs', [])))

    return dict1


def sample_systems(
    total_star_formation_in_bin,
    lookback_time_bin_size,
    lookback_time_bin_lower_edge,
    data_dict,
    config,
):
    """
    General function to handle sampling a set of systems based on
    normalized yields and a total mass of stars formed

    # TODO: we work with indices now, which might not work when we slice and dice.
    # TODO: or we should make sure the indices map back to IDs
    """

    config["logger"].warning(
        "Convolving through sampling. Using a total of {}".format(
            total_star_formation_in_bin
        )
    )

    ############
    # calculate the formation yield of all the systems
    formation_yield = (
        total_star_formation_in_bin
        * data_dict["yield_rate"]
        * config["yield_rate_unit"]
    )
    # print("formation_yield", formation_yield)

    #
    all_indices = np.arange(len(data_dict["yield_rate"]))

    ############
    # select those that have > 1:
    integer_formations = np.array(np.floor(formation_yield), dtype=int)
    # print("integer_formations", integer_formations)

    # select the remainder
    fractional_formations = formation_yield - integer_formations
    # print("fractional_formations", fractional_formations)

    # take a random set to sample the fractional formations
    random_chance = np.random.random(fractional_formations.shape)
    # print("random_chance", random_chance)

    fractional_formations_sampled = random_chance < fractional_formations
    # print("fractional_formations_sampled", fractional_formations_sampled)

    # Sample the indices
    integer_formation_indices = np.repeat(all_indices, integer_formations)
    # print("integer_formation_indices", integer_formation_indices)

    fractional_formation_indices = all_indices[fractional_formations_sampled]
    # print("fractional_formation_indices", fractional_formation_indices)

    combined_indices = np.concatenate(
        [integer_formation_indices, fractional_formation_indices]
    )
    # print("combined_indices", combined_indices)

    # print("data_dict", data_dict)

    ############
    # Make a copy of the data dict and select everything using the combined indices
    data_dict_sampled_systems = {
        data_key: data_dict[data_key][combined_indices] for data_key in data_dict.keys()
    }

    # Assign random formation times (of system)
    sampled_formation_lookback_times = (
        np.random.random(size=len(combined_indices)) * lookback_time_bin_size
    ) + lookback_time_bin_lower_edge
    # print("sampled_formation_lookback_times", sampled_formation_lookback_times)

    # add to data_dict
    data_dict_sampled_systems["formation_lookback_times"] = (
        sampled_formation_lookback_times
    )

    #
    config["logger"].warning("Sampled {} systems.".format(len(combined_indices)))

    return data_dict_sampled_systems


def sample_systems_main(
    config,
    sfr_dict,
    job_dict,
    convolution_instruction,
    data_dict,
    star_formation_rate_in_lookback_time_bin,
    lookback_time_bin_size,
    lookback_time_bin_lower_edge,
    metallicity_distribution_at_lookback_time=None,
    metallicity_bins=None,
    position_sampling_function=None,
):
    """
    Function that handles sampling systems at a particular lookback time.

    Optionally, we handle sampling in metallicity. `metallicity_distribution_at_lookback_time` is expected to contain (dP/dz)*dz

    NOTE: data_dict should contain IDs of some sort
    """

    total_star_formation_in_lookback_time_bin = (
        star_formation_rate_in_lookback_time_bin * lookback_time_bin_size
    )

    #
    config["logger"].warning(
        "Main sampling through convolution. Will sample systems according to their normalized yield and the total mass formed in stars."
    )
    config["logger"].warning(
        "Lower time bin {} upper time bin {} total mass formed {}".format(
            lookback_time_bin_lower_edge,
            lookback_time_bin_lower_edge + lookback_time_bin_size,
            total_star_formation_in_lookback_time_bin,
        )
    )

    ############
    # Method 1: no metallicity dependence
    if metallicity_distribution_at_lookback_time is None:
        config["logger"].warning(
            "Convolution sampling with absolute rate only (not using metallicity)"
        )

        sampled_data_dict = sample_systems(
            total_star_formation_in_bin=total_star_formation_in_lookback_time_bin,
            data_dict=data_dict,
            lookback_time_bin_size=lookback_time_bin_size,
            lookback_time_bin_lower_edge=lookback_time_bin_lower_edge,
            config=config,
        )

    ############
    # Method 2: metallicity dependence
    else:

        if metallicity_bins is None:
            raise ValueError("Please provide metallicity bins")
        config["logger"].warning("Convolution sampling using metallicity distributions")

        #
        sampled_data_dict = {}

        # loop over the metallicities
        for metallicity_i, metallicity_weight in enumerate(
            metallicity_distribution_at_lookback_time
        ):
            # print("metallicity_i, metallicity_weight", metallicity_i, metallicity_weight)
            total_star_formation_in_lookback_time_bin_in_metallicity_bin = (
                total_star_formation_in_lookback_time_bin * metallicity_weight
            )

            #
            lower_edge_metallicity_bin, upper_edge_metallicity_bin = (
                metallicity_bins[metallicity_i],
                metallicity_bins[metallicity_i + 1],
            )

            config["logger"].warning(
                "Metallicity {} lower bin edge {} upper bin edge {}".format(
                    metallicity_i,
                    lower_edge_metallicity_bin,
                    upper_edge_metallicity_bin,
                )
            )
            config["logger"].warning(
                "Total mass formed in current metallicity bin {}".format(
                    total_star_formation_in_lookback_time_bin_in_metallicity_bin
                )
            )

            # Select those systems that match the current metallicity bin
            matching_metallicity_indices = np.where(
                (data_dict["metallicity"] > lower_edge_metallicity_bin)
                & (data_dict["metallicity"] <= upper_edge_metallicity_bin)
            )
            # print("matching_metallicity_indices", matching_metallicity_indices)

            # Create data dict for those that match this metallicity
            metallicity_matching_data_dict = {
                data_key: data_dict[data_key][matching_metallicity_indices]
                for data_key in data_dict.keys()
            }
            # print("metallicity_matching_data_dict", metallicity_matching_data_dict)

            #
            metallicity_sampled_data_dict = sample_systems(
                total_star_formation_in_bin=total_star_formation_in_lookback_time_bin_in_metallicity_bin,
                data_dict=metallicity_matching_data_dict,
                lookback_time_bin_size=lookback_time_bin_size,
                lookback_time_bin_lower_edge=lookback_time_bin_lower_edge,
                config=config,
            )
            # print("metallicity_sampled_data_dict", metallicity_sampled_data_dict)

            # combine this with the previous dict
            sampled_data_dict = combine_dicts_with_numpy_array_entries(
                sampled_data_dict, metallicity_sampled_data_dict
            )
            # print("combined sampled_data_dict", sampled_data_dict)
            print("\n")

    ###########
    # TODO: move to separate function
    if position_sampling_function is not None:

        # Construct what parameters are available for the extra function
        available_parameters = {
            "config": config,
            "job_dict": job_dict,
            "sfr_dict": sfr_dict,
            "sfr_dict": sfr_dict,
            "data_dict": data_dict,
            "time_value": job_dict["convolution_time_bin_center"],
            "convolution_instruction": convolution_instruction,
            **convolution_instruction.get(
                "position_sampling_function_extra_parameters", {}
            ),  #
        }

        # TODO: abstract this and the extra weights calculation into a general function that selects and calls a function

        # Make sure we extract the correct things from the available parameters
        position_sampling_function_args = extract_arguments(
            func=position_sampling_function,
            arg_dict=available_parameters,
        )

        #
        config["logger"].debug(
            "Calculating positions using function {} and arguments {}".format(
                convolution_instruction["position_sampling_function"].__name__,
                position_sampling_function_args,
            )
        )

        # Call position function
        positions = position_sampling_function(**position_sampling_function_args)
        if positions is None:
            raise ValueError(
                "The position sampling function did not return a correct set of positions"
            )

        # add to dict
        # TODO: perhaps unpack into separate columns
        sampled_data_dict["positions"] = positions

    ###########
    # Sort the results on the IDs
    # print("sampled_data_dict", sampled_data_dict)

    # Sort on ID
    sorted_indices = sampled_data_dict["IDs"].argsort()

    sampled_data_dict = {
        data_key: sampled_data_dict[data_key][sorted_indices]
        for data_key in sampled_data_dict.keys()
    }
    # print("sorted sampled_data_dict", sampled_data_dict)

    ###########
    # wrap up

    # delete the normalized yield
    del sampled_data_dict["yield_rate"]

    return {"convolution_result": sampled_data_dict}


if __name__ == "__main__":

    import copy
    import json
    import os

    from syntheticstellarpopconvolve import convolve, default_convolution_config
    from syntheticstellarpopconvolve.general_functions import temp_dir

    TMP_DIR = temp_dir("code", "convolve_stochastically", clean_path=True)

    import h5py
    import pandas as pd

    ##################
    # Testing method without metallicity distribution

    time_start = time.time()

    #
    lookback_time_index = 5
    scale_factor = 5e-9
    size = 100

    # have some starformation array
    lookback_time_bin_edges = (np.arange(0, 10, 1) * u.Gyr).to(u.yr)
    starformation_rate_array = (
        0.25 * np.ones(lookback_time_bin_edges.shape[0] - 1) * u.Msun / u.yr
    )  # example of a constant star-formation rate. this could be anything of course.
    # print(starformation_array)

    bin_sizes = np.diff(lookback_time_bin_edges)
    # print(bin_sizes)

    #
    total_star_formation_at_lookback_times = starformation_rate_array * bin_sizes
    # print(total_star_formation_at_lookback_times)

    #
    normalized_yield_array = scale_factor * np.random.random(size=size)
    # print("normalized_yield_array", normalized_yield_array)

    #
    data_dict = {}
    data_dict["normalized_yield_array"] = normalized_yield_array
    data_dict["IDs"] = np.array([uuid.uuid4().hex for _ in range(size)])
    # print(data_dict)

    # #
    # sample_systems_main(
    #     total_star_formation_in_lookback_time_bin=total_star_formation_at_lookback_times[
    #         lookback_time_index
    #     ],
    #     data_dict=data_dict,
    #     lookback_time_bin_lower_edge=lookback_time_bin_edges[lookback_time_index],
    #     lookback_time_bin_size=bin_sizes[lookback_time_index],
    #     metallicity_distribution_at_lookback_time=None,
    #     metallicity_bins=None,
    # )

    ########################
    # use proper setup for convolution

    # create file
    input_hdf5_filename = os.path.join(TMP_DIR, "input_hdf5.h5")
    output_hdf5_filename = os.path.join(TMP_DIR, "output_hdf5.h5")
    input_hdf5_file = h5py.File(input_hdf5_filename, "w")

    # Create groups main
    input_hdf5_file.create_group("input_data")
    input_hdf5_file.create_group("config")

    # add group for events
    input_hdf5_file.create_group("input_data/events")

    # Write population config to file
    input_hdf5_file.create_dataset("config/population", data=json.dumps({}))

    # close
    input_hdf5_file.close()

    # load into pd
    df = pd.DataFrame.from_dict(data_dict)

    # store the data frame in the hdf5file
    df.to_hdf(input_hdf5_filename, key="input_data/events/stochastic_example")

    #
    convolution_config = copy.copy(default_convolution_config)
    convolution_config["input_filename"] = input_hdf5_filename
    convolution_config["output_filename"] = output_hdf5_filename
    convolution_config["tmp_dir"] = TMP_DIR
    convolution_config["redshift_interpolator_data_output_filename"] = os.path.join(
        TMP_DIR, "interpolator_dict.p"
    )
    convolution_config["multiply_by_time_binsize"] = False

    ###
    # convolution instructions
    convolution_config["convolution_instructions"] = [
        {
            "input_data_type": "event",
            "convolution_type": "sample",
            "input_data_name": "stochastic_example",
            "output_data_name": "stochastic_example",
            "ignore_metallicity": True,
            "data_column_dict": {
                # required
                "IDs": "IDs",
                "yield_rate": "normalized_yield_array",
                # # optional*
                # 'metallicity': 'metallicity',
            },
        },
    ]

    #
    convolution_config["time_type"] = "lookback_time"
    convolution_config["convolution_lookback_time_bin_edges"] = (
        np.arange(2, 4, 0.5) * u.Gyr
    )

    # construct the sfr-dict (NOTE: this uses absolute SFR, not metallicity dependent)
    sfr_dict = {}
    sfr_dict["lookback_time_bin_edges"] = (np.arange(0, 10, 1) * u.Gyr).to(u.yr)
    sfr_dict["starformation_rate_array"] = (
        0.25 * np.ones(sfr_dict["lookback_time_bin_edges"].shape[0] - 1) * u.Msun / u.yr
    )  # example of a constant star-formation rate. this could be anything of course.

    # store
    convolution_config["SFR_info"] = sfr_dict

    input_hdf5_file = h5py.File(input_hdf5_filename, "r")

    # convolve
    convolve(config=convolution_config)

    print("finished convolution")
    # Show some of the content
    with h5py.File(convolution_config["output_filename"], "r") as output_hdf5_file:
        # print(output_hdf5_file["output_data/"].keys())
        # print(output_hdf5_file["output_data/event/"].keys())
        # print(output_hdf5_file["output_data/event/stochastic_example/"].keys())
        # print(
        #     output_hdf5_file[
        #         "output_data/event/stochastic_example/stochastic_example"
        #     ].keys()
        # )
        # print(
        #     output_hdf5_file[
        #         "output_data/event/stochastic_example/stochastic_example/convolved_array"
        #     ].keys()
        # )

        # print(
        #     output_hdf5_file[
        #         "output_data/event/stochastic_example/stochastic_example/convolved_array/2.25 Gyr"
        #     ].keys()
        # )

        print(
            output_hdf5_file[
                "output_data/event/stochastic_example/stochastic_example/convolved_array/2.25 Gyr"
            ]["IDs"][()]
        )

        print(
            output_hdf5_file[
                "output_data/event/stochastic_example/stochastic_example/convolved_array/2.25 Gyr"
            ]["formation_lookback_times"][()]
        )

    # import astropy.units as u
    # import legwork as lw
    # import numpy as np

    # N = 100
    # m_1 = np.random.uniform(1, 10, N) * u.Msun
    # m_2 = np.random.rand(N) * m_1
    # dist = np.random.uniform(1, 30, N) * u.kpc
    # f_orb_i = 10**(np.random.uniform(-5, -2, N)) * u.Hz
    # ecc_i = np.random.rand(N)

    # sources = lw.source.Source(m_1=m_1, m_2=m_2, ecc=ecc_i, f_orb=f_orb_i, dist=dist,
    #                            interpolate_g=N > 1000)

    # t_evol = np.random.uniform(0.1, 1, N) * u.Myr

    # sources.evolve_sources(t_evol)

    # print(sources.f_orb, sources.ecc)

    quit()

    ##################
    # Testing method with metallicity distribution

    #
    data_dict["metallicity"] = np.random.random(size=size)

    # print("data_dict['metallicity']", data_dict['metallicity'])

    metallicity_distribution_at_lookback_time = np.array([0.25, 0.25, 0.25, 0.25])
    metallicity_bins = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

    # time_start_convolution = time.time()

    # #
    # sampled_data_dict = sample_systems_main(
    #     total_star_formation_in_lookback_time_bin=total_star_formation_at_lookback_times[
    #         lookback_time_index
    #     ],
    #     data_dict=data_dict,
    #     lookback_time_bin_lower_edge=lookback_time_bin_edges[lookback_time_index],
    #     lookback_time_bin_size=bin_sizes[lookback_time_index],
    #     metallicity_distribution_at_lookback_time=metallicity_distribution_at_lookback_time,
    #     metallicity_bins=metallicity_bins,
    # )

    # time_end_convolution = time.time()

    # print("Total time: {:.2E}".format(time_end_convolution - time_start))
    # print(
    #     "Total time convolution: {:.2E}".format(
    #         time_end_convolution - time_start_convolution
    #     )
    # )
    # print(
    #     "Fractional time convolution: {:.2E}".format(
    #         (time_end_convolution - time_start_convolution)
    #         / (time_end_convolution - time_start)
    #     )
    # )

    # print(sampled_data_dict)

    #################
    # formal convolve with metallicity

    # create file
    input_hdf5_filename = os.path.join(TMP_DIR, "input_hdf5.h5")
    output_hdf5_filename = os.path.join(TMP_DIR, "output_hdf5.h5")
    input_hdf5_file = h5py.File(input_hdf5_filename, "w")

    # Create groups main
    input_hdf5_file.create_group("input_data")
    input_hdf5_file.create_group("config")

    # add group for events
    input_hdf5_file.create_group("input_data/events")

    # Write population config to file
    input_hdf5_file.create_dataset("config/population", data=json.dumps({}))

    # close
    input_hdf5_file.close()

    # load into pd
    df = pd.DataFrame.from_dict(data_dict)

    # store the data frame in the hdf5file
    df.to_hdf(input_hdf5_filename, key="input_data/events/stochastic_example")

    #
    convolution_config = copy.copy(default_convolution_config)
    convolution_config["input_filename"] = input_hdf5_filename
    convolution_config["output_filename"] = output_hdf5_filename
    convolution_config["tmp_dir"] = TMP_DIR
    convolution_config["redshift_interpolator_data_output_filename"] = os.path.join(
        TMP_DIR, "interpolator_dict.p"
    )
    convolution_config["multiply_by_time_binsize"] = False

    ###
    # convolution instructions
    convolution_config["convolution_instructions"] = [
        {
            "input_data_type": "event",
            "convolution_type": "sample",
            "input_data_name": "stochastic_example",
            "output_data_name": "stochastic_example",
            "ignore_metallicity": True,
            "data_column_dict": {
                # required
                "IDs": "IDs",
                "yield_rate": "normalized_yield_array",
                "metallicity": "metallicity",
                # # optional*
                # 'metallicity': 'metallicity',
            },
        },
    ]

    #
    convolution_config["time_type"] = "lookback_time"
    convolution_config["convolution_lookback_time_bin_edges"] = (
        np.arange(2, 4, 0.5) * u.Gyr
    )

    # construct the sfr-dict (NOTE: this uses absolute SFR, not metallicity dependent)
    sfr_dict = {}
    sfr_dict["lookback_time_bin_edges"] = (np.arange(0, 10, 1) * u.Gyr).to(u.yr)
    sfr_dict["starformation_rate_array"] = (
        0.25 * np.ones(sfr_dict["lookback_time_bin_edges"].shape[0] - 1) * u.Msun / u.yr
    )  # example of a constant star-formation rate. this could be anything of course.

    sfr_dict["metallicity_weighted_starformation_rate_array"] = (
        sfr_dict["starformation_rate_array"][:, np.newaxis]
        * metallicity_distribution_at_lookback_time[np.newaxis, :]
    )
    sfr_dict["metallicity_bin_edges"] = metallicity_bins

    # store
    convolution_config["SFR_info"] = sfr_dict

    input_hdf5_file = h5py.File(input_hdf5_filename, "r")

    # convolve
    convolve(config=convolution_config)

    print("finished convolution")
    # Show some of the content
    with h5py.File(convolution_config["output_filename"], "r") as output_hdf5_file:
        print(output_hdf5_file["output_data/"].keys())
        print(output_hdf5_file["output_data/event/"].keys())
        print(output_hdf5_file["output_data/event/stochastic_example/"].keys())
        print(
            output_hdf5_file[
                "output_data/event/stochastic_example/stochastic_example"
            ].keys()
        )
        print(
            output_hdf5_file[
                "output_data/event/stochastic_example/stochastic_example/convolved_array"
            ].keys()
        )

        print(
            output_hdf5_file[
                "output_data/event/stochastic_example/stochastic_example/convolved_array/2.25 Gyr"
            ].keys()
        )

        print(
            output_hdf5_file[
                "output_data/event/stochastic_example/stochastic_example/convolved_array/2.25 Gyr"
            ]["IDs"][()]
        )

        print(
            output_hdf5_file[
                "output_data/event/stochastic_example/stochastic_example/convolved_array/2.25 Gyr"
            ]["formation_lookback_times"][()]
        )
