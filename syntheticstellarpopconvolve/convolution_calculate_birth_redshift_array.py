"""
Functions related to the arrays we store the
"""

import astropy.units as u
import numpy as np

from syntheticstellarpopconvolve.cosmology_utils import redshift_to_lookback_time


####
# Array functions
def calculate_origin_redshift_array(
    config,
    convolution_redshift_value,
    data_dict,
):
    """
    TODO: update text
    Function to calculate the birth redshift array for the merging/formation events based on
    the current redshift and the merger/formation time of the system.

    We do this by calculating the lookback time (which is age_of_universe(z=0) - age_of_universe(z))
    adding the merger time of the system to the lookback time, getting the birth time
    and then calculating the birth redshift from the birth time
    """

    config["logger"].debug(
        "Calculating origin redshift of systems by converting to lookback time, adding delay time and converting back to redshift."
    )

    # With the current redshift, we calculate the lookback time, subtract the merger time and formation time and turn back into
    current_lookback_value = redshift_to_lookback_time(
        convolution_redshift_value, cosmology=config["cosmology"]
    )

    # Calculate lookback time of first starformation (which is the same as the upper redshift of the interpolator)
    lookback_time_of_first_starformation = redshift_to_lookback_time(
        config["redshift_interpolator_max_redshift"],
        cosmology=config["cosmology"],
    )

    # Calculate the lookback time of the event, given the delay time of the event (i.e. duration between birth and event-type) and the current time.
    origin_lookback_time_values_in_gyr = (
        current_lookback_value.to(u.yr) + data_dict["delay_time"].to(u.yr)
    ).to(u.Gyr)

    # Get the indices where the event falls inside the correct starformation time range
    indices_within_first_starformation = (
        origin_lookback_time_values_in_gyr < lookback_time_of_first_starformation
    )
    indices_outside_first_starformation = (
        origin_lookback_time_values_in_gyr >= lookback_time_of_first_starformation
    )

    # Create redshift values array of the event
    origin_redshift_values = np.ones(data_dict["delay_time"].shape)
    origin_redshift_values[indices_within_first_starformation] = config[
        "interpolators"
    ]["lookback_time_to_redshift_interpolator"](
        origin_lookback_time_values_in_gyr[indices_within_first_starformation]
    )
    origin_redshift_values[indices_outside_first_starformation] = -1

    #
    return origin_redshift_values
