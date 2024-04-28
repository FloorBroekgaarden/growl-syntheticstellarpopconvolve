"""
Main entry point to the convolution code. This code handles passing choosing the correct code to do the convolution with.

TODO: make wrappers to handle timing logging to debug.
TODO: allow usage of astropy units
"""

from syntheticstellarpopconvolve.check_convolution_input_file import (
    check_convolution_input_file,
)
from syntheticstellarpopconvolve.convolution_check_config import (
    check_convolution_config,
)
from syntheticstellarpopconvolve.convolve_populations import convolve_populations
from syntheticstellarpopconvolve.extract_population_settings import (
    extract_population_settings,
)
from syntheticstellarpopconvolve.prepare_output_file import prepare_output_file
from syntheticstellarpopconvolve.prepare_redshift_interpolator import (
    prepare_redshift_interpolator,
)
from syntheticstellarpopconvolve.update_convolution_config import (
    update_convolution_config,
)


def convolve(config):  # DH0001
    """
    Main function to run the convolution

    Generally the functions below require that there exists some
    """

    #
    config["logger"].debug("Starting convolution")

    ##############################################
    # Setup phase

    ###########
    # Check the config to see if the configuration for the convolution code is correct and not missing anything.
    check_convolution_config(config=config)

    ###########
    # Update the config with some extra calculated stuff
    update_convolution_config(config=config)

    ###########
    # Check the input file
    check_convolution_input_file(config=config)

    ###########
    # Copy the input file and
    prepare_output_file(config=config)

    # ###########
    # # Calculate SFR information and add to hdf5 file
    # store_sfr_info(config=config)

    ###########
    # Extract some information to store in the config
    config = extract_population_settings(config=config)

    ###########
    # Calculate SFR information and add to hdf5 file
    config = prepare_redshift_interpolator(config=config)

    ##############################################
    # Convolution phase
    convolve_populations(config=config)

    ##############################################
    # Cleanup phase
    # TODO: implement cleanup

    #
    config["logger"].debug("Convolution finished")
