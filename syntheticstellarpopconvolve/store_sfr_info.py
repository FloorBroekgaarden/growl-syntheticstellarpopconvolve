"""
Function to calculate the SFR information and store it in the output file
"""

# import h5py


# def extract_sfr_info_from_file(SFR_file):
#     """
#     Function to extract the SFR info from the SFR file
#     """

#     raise NotImplementedError()


# def store_sfr_info(config):
#     """
#     Function to calculate the SFR information and store it in the dataframe.

#     TODO: allow user to provide the something like units. Things can be mass per year. Mass per year per volume. Mass per year per surface density etc.
#     """

#     # if config["use_SFR_file"]:

#     #     # TODO: implement method to readout SFR history files

#     #     (
#     #         metallicity_weighted_starformation_array,
#     #         metallicity_fraction_array,
#     #         starformation_array,
#     #     ) = extract_sfr_info_from_file(config["SFR_file"])

#     # else:
#     # TODO: the below method is based on providing functional forms. We also
#     # want to allow SFR history files of a certain format probably. so we should
#     # accomodate that

#     # Calculate SFR information
#     (
#         metallicity_weighted_starformation_array,
#         metallicity_fraction_array,
#         starformation_array,
#     ) = generate_metallicity_sfr_array(
#         config=config,
#         star_formation_rate_time_distribution_bin_centers=config[
#             "star_formation_rate_distribution_time_bin_centers"
#         ],
#         metallicity_centers=config["convolution_metallicity_bin_centers"],
#     )
#     metallicity_weighted_starformation_array = (
#         metallicity_weighted_starformation_array.T
#     )

#     #
#     # metallicity_centers = config["metallicity_centers"]
#     convolution_metallicity_bin_edges = config["convolution_metallicity_bin_edges"]

#     # # TOOD: calculate starformation rate history. This returns a 1-d distribution with only a rate.
#     # calculate_starformation_rate()

#     # # TODO: calculate normalized metallicity distribution
#     # calculate_metallicity_distribution()

#     #############
#     # Store the information
#     with h5py.File(config["output_filename"], "a") as output_hdf5file:

#         sfr_grp = output_hdf5file["starformation"]

#         # store time info
#         sfr_grp.create_dataset(
#             "convolution_time_bin_centers", data=config["convolution_time_bin_centers"]
#         )
#         sfr_grp.create_dataset(
#             "convolution_time_bin_edges", data=config["convolution_time_bin_edges"]
#         )

#         # store metallicity info
#         # sfr_grp.create_dataset("metallicity_centers", data=metallicity_centers)
#         sfr_grp.create_dataset(
#             "convolution_metallicity_bin_edges", data=convolution_metallicity_bin_edges
#         )

#         # Write to output file
#         sfr_grp.create_dataset(
#             "metallicity_weighted_starformation_array",
#             data=metallicity_weighted_starformation_array,
#         )
#         sfr_grp.create_dataset(
#             "metallicity_fraction_array", data=metallicity_fraction_array
#         )
#         sfr_grp.create_dataset("starformation_array", data=starformation_array)
#         sfr_grp.create_dataset(
#             "starformation_unit",
#             data=metallicity_weighted_starformation_array.unit.to_string(),
#         )
