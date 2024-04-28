"""
Main file to handle the convolution of populations

TODO: when units are passed back to we need to store them in the meta-data
"""

import json
import multiprocessing
import os
import pickle

import h5py
import setproctitle

from syntheticstellarpopconvolve.convolution_general_functions import (
    JsonCustomEncoder,
    generate_group_name,
    get_tmp_dir,
    pad_function,
)
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
from syntheticstellarpopconvolve.store_redshift_shell_info import (
    store_redshift_shell_info,
)

CONVOLUTION_FUNCTION_DICT = {
    "event": event_convolution_function,
    "ensemble": ensemble_convolution_function,
    "custom": custom_convolution_function,
}


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
    """

    #
    config["logger"].debug("Padding SFR dict")

    max_pad = 1.0e13

    ##########
    # pad lookback time/redshift array
    if config["time_type"] == "lookback_time":
        #
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
    else:
        raise ValueError("Invalid time-type")

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
    TODO
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
    if config["write_to_hdf5"]:
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
            for file in sorted_content_dir:

                # Load pickled data
                full_path = os.path.join(tmp_dir, file)
                with open(full_path, "rb") as picklefile:
                    data = pickle.load(picklefile)

                # Store payload in grp
                config["logger"].debug(
                    "Storing convolution results of bin-center {}".format(
                        str(data["convolution_time_bin_center"])
                    )
                )
                grp.create_dataset(
                    "convolved_array/{}".format(
                        str(data["convolution_time_bin_center"])
                    ),
                    data=data["convolution_result"],
                )

                # if cleaned ensemble is included, add that too
                if "stripped_ensemble" in data.keys():
                    config["logger"].debug("Storing stripped ensemble")

                    grp.create_dataset(
                        "stripped_ensemble", data=json.dumps(data["stripped_ensemble"])
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
        convolution_time_bin_center = job_dict["convolution_time_bin_center"]
        convolution_instruction = job_dict["convolution_instruction"]
        data_dict = job_dict["data_dict"]

        ##########
        # Set up output dict
        output_dict = {}

        ##########
        #
        config["logger"].debug(
            "Worker {}: convolution_time_bin_center: {}: Calculating {} {} rates".format(
                worker_ID,
                convolution_time_bin_center,
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
            convolution_time_bin_center=convolution_time_bin_center,
            job_dict=job_dict,
            config=config,
            convolution_instruction=convolution_instruction,
            data_dict=data_dict,
        )

        # Construct dictionary that is stored in the pickle files
        output_dict["convolution_time_bin_center"] = convolution_time_bin_center
        output_dict["convolution_instruction"] = convolution_instruction
        output_dict = {**output_dict, **convolution_result_dict}

        #
        with open(
            os.path.join(
                job_dict["output_dir"], "{}.p".format(convolution_time_bin_center)
            ),
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
    """

    # Fill the queue with centres
    for convolution_bin_number, (
        convolution_bin_center,
        convolution_bin_size,
    ) in enumerate(
        zip(
            config["convolution_time_bin_centers"], config["convolution_time_bin_sizes"]
        )
    ):
        # Set up job dict
        job_dict = {
            "job_number": convolution_bin_number,
            "convolution_time_bin_center": convolution_bin_center,
            "convolution_time_bin_size": convolution_bin_size,
            "convolution_time_bin_number": convolution_bin_number,
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

        ######
        # Update sfr dict: pads dict, adds redshift shells
        sfr_dict = update_sfr_dict(sfr_dict=sfr_dict, config=config)

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
