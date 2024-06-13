"""
Main file to handle the convolution of populations

TODO: ma
"""

import json
import multiprocessing
import os
import pickle

import h5py
import numpy as np
import setproctitle

from syntheticstellarpopconvolve.convolve_custom_data import (
    custom_convolution_function,
    extract_custom_data,
)
from syntheticstellarpopconvolve.convolve_ensembles import (
    ensemble_convolution_function,
    extract_ensemble_data,
)
from syntheticstellarpopconvolve.convolve_events import (
    event_convolution_function,
    extract_event_data,
)
from syntheticstellarpopconvolve.general_functions import (
    JsonCustomEncoder,
    generate_group_name,
    get_tmp_dir,
)

CONVOLUTION_FUNCTION_DICT = {
    "event": event_convolution_function,
    "ensemble": ensemble_convolution_function,
    "custom": custom_convolution_function,
    #  "event_sample",
}


def pre_multiprocessing(config, convolution_instruction, sfr_dict):  # DH0001
    """
    TODO
    """

    ########
    # get groupname
    groupname, elements = generate_group_name(
        convolution_instruction=convolution_instruction, sfr_dict=sfr_dict
    )

    ########
    # Apply correct structure in hdf5 file
    with h5py.File(config["output_filename"], "a") as output_hdf5file:
        ########
        # Create output data group
        config["logger"].debug("Creating output data groups '{}'".format(groupname))

        #
        if "output_data" not in output_hdf5file.keys():
            output_hdf5file.create_group("output_data")

        # Create further structure of data group
        for depth in range(len(elements)):
            output_hdf5file["output_data"].create_group("/".join(elements[: depth + 1]))

        ########
        # store SFR dict
        if "name" in sfr_dict:
            group_ = "output_data/{}".format(sfr_dict["name"])
        else:
            group_ = "output_data"

        config["logger"].debug(
            "Storing SFR dict in attribute of group '{}'".format(group_)
        )

        #
        output_hdf5file[group_].attrs["SFR_info"] = json.dumps(
            sfr_dict, cls=JsonCustomEncoder
        )

    ########
    # create tmp dir
    tmp_dir = get_tmp_dir(
        config=config,
        convolution_instruction=convolution_instruction,
        sfr_dict=sfr_dict,
    )
    os.makedirs(tmp_dir, exist_ok=True)


def post_multiprocessing(config, convolution_instruction, sfr_dict):  # DH0001
    """
    TODO write each array as a sub group in the convolution
    results. instead of dumping the data blindly within
    convolved_array/ lets rename it to convolution_results and under
    that umbrella we throw different kinds of data

    data types:
    - yield (integration, events and ensemble): SFR weighted probabilities of each system
    - stripped_ensemble (integration, ensembe): Ensemble with its endpoints stripped off. Will only be stored in the first one and should be used to re-construct the other results
    - sampled_IDs: (sampling, events): IDs of sampled systems
    - sampled_birth_times: (sampling, events): Assigned birth-times of sampled systems.
    - sampled_positions: (sampling, events): sampled positions. Can be multi-d.

    We can automatically store these and update some of the meta-data
    """

    #################
    # Put pickle data in the hdf5 file
    tmp_dir = get_tmp_dir(
        config=config,
        convolution_instruction=convolution_instruction,
        sfr_dict=sfr_dict,
    )

    ########
    # Write results to output file
    if not config["write_to_hdf5"]:
        return

    # Get groupname
    groupname, _ = generate_group_name(
        convolution_instruction=convolution_instruction, sfr_dict=sfr_dict
    )
    full_groupname = "output_data/" + groupname

    with h5py.File(config["output_filename"], "a") as output_hdf5file:
        config["logger"].debug("Writing results to {}".format(full_groupname))

        # Readout group
        grp = output_hdf5file[full_groupname]

        ###########
        # loop over all files in the pickle
        content_dir = os.listdir(tmp_dir)

        sorted_content_dir = sorted(
            content_dir,
            key=lambda x: float(".".join(x.split(".")[:-1]).split(" ")[0]),
        )
        for pickle_file in sorted_content_dir:

            # Load pickled data
            full_path = os.path.join(tmp_dir, pickle_file)
            with open(full_path, "rb") as picklefile:
                data = pickle.load(picklefile)

            ##########
            # Unpack
            if "convolution_result" in data.keys():
                convolution_result = data["convolution_result"]
            else:
                raise ValueError("No convolution result present in the data")

            # Create group
            current_time_bin_grp = grp.create_group(
                "convolved_array/{}".format(str(data["bin_center"]))
            )

            # Store payload in grp
            config["logger"].debug(
                "Storing convolution results of bin-center {}".format(
                    str(data["bin_center"])
                )
            )

            ############
            # Store different kinds of output

            units_to_store = {}

            # yield output. From integration-based event and ensemble convolution
            if "yield" in convolution_result.keys():
                config["logger"].debug("Storing yield")

                #
                yield_value = convolution_result["yield"].value
                yield_unit = convolution_result["yield"].unit

                current_time_bin_grp.create_dataset("yield", data=yield_value)

                # store unit and description in meta-data
                units_to_store["yield"] = yield_unit

            # stripped ensemble output. From integration-based ensemble convolution
            if "stripped_ensemble" in convolution_result.keys():
                config["logger"].debug("Storing stripped ensemble")

                #
                stripped_ensemble = convolution_result["stripped_ensemble"]

                current_time_bin_grp.create_dataset(
                    "stripped_ensemble", data=json.dumps(stripped_ensemble)
                )

                # TODO: store description

            # ID output. From sampling-based event convolution
            if "IDs" in convolution_result.keys():
                config["logger"].debug("Storing IDs")

                #
                IDs = convolution_result["IDs"]
                IDs = IDs.astype("S")

                current_time_bin_grp.create_dataset("IDs", data=IDs)

                # TODO: store description

            # formation lookback-times output. From sampling-based event convolution
            if "formation_lookback_times" in convolution_result.keys():
                config["logger"].debug("Storing formation lookback-times")

                #
                formation_lookback_times = convolution_result[
                    "formation_lookback_times"
                ]
                formation_lookback_times_value = formation_lookback_times.value
                formation_lookback_times_unit = formation_lookback_times.unit

                current_time_bin_grp.create_dataset(
                    "formation_lookback_times", data=formation_lookback_times_value
                )

                # TODO: store unit and description
                # store unit and description in meta-data
                units_to_store["formation_lookback_times"] = (
                    formation_lookback_times_unit
                )

            # positions output. From sampling-based event convolution
            if "positions" in convolution_result.keys():
                config["logger"].debug("Storing positions")

                current_time_bin_grp.create_dataset(
                    "positions", data=convolution_result["positions"]
                )

                # TODO: store description

            # store attributes
            current_time_bin_grp.attrs["units"] = json.dumps(
                units_to_store, cls=JsonCustomEncoder
            )

            # remove the pickled file
            if config["remove_pickle_files"]:
                os.remove(full_path)


def convolution_job_worker(job_queue, worker_ID, config):  # DH0001
    """
    Function that handles running the job
    """

    setproctitle.setproctitle(
        "convolution multiprocessing worker process {}".format(worker_ID)
    )

    # Get items from the job_queue
    for job_dict in iter(job_queue.get, "STOP"):
        #########
        # Stopping or working
        if job_dict == "STOP":
            return None

        # Unpack info
        bin_center = job_dict["bin_center"]
        convolution_instruction = job_dict["convolution_instruction"]

        data_dict = job_dict["data_dict"]

        ##########
        # Set up output dict
        output_dict = {}

        ##########
        #
        config["logger"].debug(
            "Worker {}: {} bin center: {}: Calculating {} {} rates".format(
                worker_ID,
                (
                    "convolution time"
                    if convolution_instruction["convolution_type"] == "integrate"
                    else "starformation time"
                ),
                bin_center,
                convolution_instruction["input_data_type"],
                convolution_instruction["input_data_name"],
            )
        )

        # -----------------------------------------------------------------------
        # Handle the convolution depending on which type of data exists. They
        # all contain the same structure.
        #
        # The resulting dictionary contains at
        # least the results of the convolution (i.e. an array of 'rates' or
        # total yields), and potentially more, depending on what each function
        # returns. ensemble convolution for example can return a stripped
        # ensemble
        #

        # run conolution with the appropriate function
        convolution_result_dict = CONVOLUTION_FUNCTION_DICT[
            convolution_instruction["input_data_type"]
        ](
            bin_center=bin_center,
            job_dict=job_dict,
            config=config,
            convolution_instruction=convolution_instruction,
            data_dict=data_dict,
        )

        # Construct dictionary that is stored in the pickle files
        output_dict["bin_center"] = bin_center
        output_dict["convolution_instruction"] = convolution_instruction
        output_dict = {**output_dict, **convolution_result_dict}

        #
        with open(
            os.path.join(job_dict["output_dir"], "{}.p".format(bin_center)),
            "wb",
        ) as f:
            pickle.dump(output_dict, f)


def convolution_queue_filler(  # DH0001
    job_queue,
    num_cores,
    config,
    sfr_dict,
    convolution_instruction,
    data_dict,
):
    """
    Function to handle filling the queue for the multiprocessing

    When the convolution instruction is a sampling-based convolution,
    we use forward convolution, which loops over starformation bins
    rather than convolution bins
    """

    ######
    # Determine bins to loop over (integrate = backward conv, sampling = forward conv)
    if convolution_instruction["convolution_type"] == "integrate":
        zipped_bin_data = zip(
            config["convolution_time_bin_centers"], config["convolution_time_bin_sizes"]
        )
    elif convolution_instruction["convolution_type"] == "sample":

        # TODO:  and put into sfr dict check and update.
        # TODO: generalize this to also use redshift
        starformation_time_bin_sizes = np.diff(sfr_dict["lookback_time_bin_edges"])
        starformation_time_bin_centers = (
            sfr_dict["lookback_time_bin_edges"][1:]
            + sfr_dict["lookback_time_bin_edges"][:-1]
        ) / 2

        # # TODO: move to sfr dict checking
        # starformation_bin_sizes = np.diff(sfr_dict["starformation_rate_array"])
        # starformation_bin_centers = (
        #     sfr_dict["starformation_rate_array"][1:]
        #     + sfr_dict["starformation_rate_array"][:-1]
        # ) / 2

        zipped_bin_data = zip(
            starformation_time_bin_centers, starformation_time_bin_sizes
        )
    else:
        raise ValueError("convolution type not supported")

    ######
    # Fill the queue with centres
    for bin_number, (
        bin_center,
        bin_size,
    ) in enumerate(zipped_bin_data):
        # Set up job dict
        job_dict = {
            "job_number": bin_number,
            "bin_center": bin_center,
            "bin_size": bin_size,
            "bin_number": bin_number,
            "sfr_dict": sfr_dict,
            "convolution_instruction": convolution_instruction,
            "data_dict": data_dict,
            "output_dir": get_tmp_dir(
                config=config,
                convolution_instruction=convolution_instruction,
                sfr_dict=sfr_dict,
            ),
        }

        #
        config["logger"].debug("job {} in the queue".format(job_dict["job_number"]))

        # Put job in queue
        job_queue.put(job_dict)

    # Signal stop to workers
    config["logger"].debug("Sending job termination signals")
    for _ in range(num_cores):
        job_queue.put("STOP")


def generate_data_dict(config, convolution_instruction):
    """
    Function to generate the data dict.
    """

    extractor_functions = {
        "event": extract_event_data,
        "ensemble": extract_ensemble_data,
        "custom": extract_custom_data,
    }

    #
    config["logger"].debug(
        "Generating data_dict using the extractor function for {}: {}".format(
            convolution_instruction["input_data_type"],
            extractor_functions[convolution_instruction["input_data_type"]].__name__,
        )
    )

    #
    config, data_dict, convolution_instruction = extractor_functions[
        convolution_instruction["input_data_type"]
    ](config=config, convolution_instruction=convolution_instruction)

    return config, data_dict, convolution_instruction


def multiprocess_convolution(config, convolution_instruction, sfr_dict):  # DH0001
    """
    Main multiprocess function
    """

    ###################
    # Set up data_dict: dictionary that contains the arrays or ensembles that are required for the convolution.
    config, data_dict, convolution_instruction = generate_data_dict(
        config=config, convolution_instruction=convolution_instruction
    )

    ###################
    # Run the convolution through multiprocessing

    # Set process name
    setproctitle.setproctitle("Convolution parent process")

    # Set up the manager object that can share info between processes
    manager = multiprocessing.Manager()
    job_queue = manager.Queue(config["max_job_queue_size"])

    # Create process instances
    processes = []
    for worker_ID in range(config["num_cores"]):
        processes.append(
            multiprocessing.Process(
                target=convolution_job_worker,
                args=(job_queue, worker_ID, config),
            )
        )

    # Activate the processes
    for p in processes:
        p.start()

    # Start the system_queue and process
    convolution_queue_filler(
        job_queue=job_queue,
        num_cores=config["num_cores"],
        config=config,
        sfr_dict=sfr_dict,
        convolution_instruction=convolution_instruction,
        data_dict=data_dict,
    )

    # Join the processes to wrap up
    for p in processes:
        p.join()


def convolve_populations(config):
    """
    Main function to handle the convolution of populations
    """

    #######
    # Check if we need to provide info for the SFR loop of not
    actual_sfr_dict_loop = False
    sfr_dicts = []
    if isinstance(config["SFR_info"], dict):
        sfr_dicts = [config["SFR_info"]]
    else:
        sfr_dicts = config["SFR_info"]
        actual_sfr_dict_loop = True

    ########
    # Loop over all sfr dicts
    for sfr_dict_number, sfr_dict in enumerate(sfr_dicts):

        # provide info for sfr loop if necessary
        if actual_sfr_dict_loop:
            config["logger"].debug(
                "Handling SFR {} (number {}) ".format(sfr_dict["name"], sfr_dict_number)
            )

        ########
        # Convolution
        for convolution_instruction in config["convolution_instructions"]:

            ########
            # Pre multiprocessing calculation
            pre_multiprocessing(
                config=config,
                convolution_instruction=convolution_instruction,
                sfr_dict=sfr_dict,
            )

            ########
            # Pre multiprocessing calculation
            multiprocess_convolution(
                config=config,
                convolution_instruction=convolution_instruction,
                sfr_dict=sfr_dict,
            )

            ########
            # Post multiprocessing calculation
            post_multiprocessing(
                config=config,
                convolution_instruction=convolution_instruction,
                sfr_dict=sfr_dict,
            )
