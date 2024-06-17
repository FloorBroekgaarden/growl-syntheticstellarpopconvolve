Convolution options
===================

The following chapter contains all Population code options, along with their descriptions.

Public options
--------------

In this section we list the public options for the population code. These are meant to be changed by the user.

.. list-table:: Public options
   :widths: 25, 75
   :header-rows: 1

   * - Option
     - Description
   * - SFR_info
     - Description:
          Dictionary containing the starformation rate info. Can also be a list of dictionaries.

       Default value:
          {}
   * - check_convolution_config
     - Description:
          Flag whether to validate the configuration dictionary before running the convolution code.

       Default value:
          True

       Validation:
          All(Range(min=None, max=1, min_included=True, max_included=True, msg=None), <function Boolean at 0x7f7d14f133a0>, msg=None)
   * - convolution_instructions
     - Description:
          List of instructions for the convolution.

       Default value:
          [{}]
   * - convolution_lookback_time_bin_edges
     - Description:
          Lookback-time bin-edges used in convolution.

       Default value:
          None

       Validation:
          <function array_validation at 0x7f7d14f13280>
   * - convolution_redshift_bin_edges
     - Description:
          Redshift bin-edges used in convolution.

       Default value:
          None

       Validation:
          <function array_validation at 0x7f7d14f13280>
   * - cosmology
     - Description:
          Astropy cosmology used throughout the code.

       Default value:
          FlatLambdaCDM(name="Planck13", H0=67.77 km / (Mpc s), Om0=0.30712, Tcmb0=2.7255 K, Neff=3.046, m_nu=[0.   0.   0.06] eV, Ob0=0.048252)
   * - custom_convolution_function
     - Description:


       Default value:
          None

       Validation:
          <function callable_or_none_validation at 0x7f7d14f131f0>
   * - custom_data_extraction_function
     - Description:


       Default value:
          None

       Validation:
          <function callable_or_none_validation at 0x7f7d14f131f0>
   * - custom_rates_function
     - Description:
          Custom rate function used in the convolution.

       Default value:
          None

       Validation:
          <function callable_or_none_validation at 0x7f7d14f131f0>
   * - delay_time_default_unit
     - Description:
          Default unit used for the delay-time data. NOTE: this can be overridden in data_dict column or layer entries.

       Default value:
          yr

       Validation:
          <function unit_validation at 0x7f7d14f13040>
   * - extra_weights_function
     - Description:
          Function that calculates extra weights for each system or sub-ensemble. This functions should return a numpy array. The arguments of this function should be chosen from: 'config', 'time_value', 'convolution_instruction', 'data_dict' and the contents of 'extra_weights_function_additional_parameters'. For more explanation about this function see the convolution notebook.

       Default value:
          None

       Validation:
          <function callable_or_none_validation at 0x7f7d14f131f0>
   * - extra_weights_function_additional_parameters
     - Description:
          Additional arguments that can be accessed by the extra_weights_function.

       Default value:
          {}

       Validation:
          <class 'dict'>
   * - include_custom_rates
     - Description:
          Whether to include custom, user-specified, rates. See 'custom_rates_function'.

       Default value:
          False

       Validation:
          All(Range(min=None, max=1, min_included=True, max_included=True, msg=None), <function Boolean at 0x7f7d14f133a0>, msg=None)
   * - input_filename
     - Description:
          Full path to input hdf5 filename.

       Default value:


       Validation:
          <function existing_path_validation at 0x7f7d14f13310>
   * - logger
     - Description:
          Logger object.

       Default value:
          <Logger syntheticstellarpopconvolve.default_convolution_config (INFO)>

       Validation:
          <function logger_validation at 0x7f7d14f130d0>
   * - max_job_queue_size
     - Description:
          Max number of jobs in the multiprocessing queue for the convolution.

       Default value:
          8

       Validation:
          <class 'int'>
   * - multiply_by_time_binsize
     - Description:
          Flag to multiply the SFR value by the time-bin size. When time-type='redshift' we use the associated lookback times and the width between those to calcualte the bi.

       Default value:
          False

       Validation:
          All(Range(min=None, max=1, min_included=True, max_included=True, msg=None), <function Boolean at 0x7f7d14f133a0>, msg=None)
   * - num_cores
     - Description:
          Number of cores to use to do the convolution.

       Default value:
          1

       Validation:
          <class 'int'>
   * - output_filename
     - Description:
          Full path to output hdf5 filename.

       Default value:


       Validation:
          <class 'str'>
   * - redshift_interpolator_data_output_filename
     - Description:
          Filename for the redshift interpolator object.

       Default value:
          None

       Validation:
          <class 'str'>
   * - redshift_interpolator_force_rebuild
     - Description:
          Whether to force rebuild the redshift interpolator.

       Default value:
          False

       Validation:
          All(Range(min=None, max=1, min_included=True, max_included=True, msg=None), <function Boolean at 0x7f7d14f133a0>, msg=None)
   * - redshift_interpolator_max_redshift
     - Description:
          Minimum redshift for the redshift interpolator.

       Default value:
          50

       Validation:
          Any(<class 'float'>, <class 'int'>, msg=None)
   * - redshift_interpolator_min_redshift
     - Description:
          Minimum redshift for the redshift interpolator.

       Default value:
          0

       Validation:
          Any(<class 'float'>, <class 'int'>, msg=None)
   * - redshift_interpolator_min_redshift_if_log
     - Description:
          Minimum redshift for the redshift interpolator if using log spacing.

       Default value:
          1e-05

       Validation:
          Any(<class 'float'>, <class 'int'>, msg=None)
   * - redshift_interpolator_rebuild_when_settings_mismatch
     - Description:
          Whether to rebuild the redshift interpolator when the config of the existing one don't match with the current config.

       Default value:
          True

       Validation:
          All(Range(min=None, max=1, min_included=True, max_included=True, msg=None), <function Boolean at 0x7f7d14f133a0>, msg=None)
   * - redshift_interpolator_stepsize
     - Description:
          Stepsize for the redshift interpolation.

       Default value:
          0.001

       Validation:
          Any(<class 'float'>, <class 'int'>, msg=None)
   * - redshift_interpolator_use_log
     - Description:
          Whether to interpolate in log redshift.

       Default value:
          True

       Validation:
          All(Range(min=None, max=1, min_included=True, max_included=True, msg=None), <function Boolean at 0x7f7d14f133a0>, msg=None)
   * - remove_pickle_files
     - Description:
          Flag whether to remove all the pickle files after writing them to the main hdf5 file.

       Default value:
          True

       Validation:
          All(Range(min=None, max=1, min_included=True, max_included=True, msg=None), <function Boolean at 0x7f7d14f133a0>, msg=None)
   * - time_type
     - Description:
          Time-type used in convolution. Can be either 'redshift' or 'lookback_time'.

       Default value:
          redshift

       Validation:
          All(<class 'str'>, In(['redshift', 'lookback_time']), msg=None)
   * - tmp_dir
     - Description:
          Target directory for the tmp files.

       Default value:
          /tmp

       Validation:
          <class 'str'>
   * - write_to_hdf5
     - Description:
          Whether to write the pickle-files from the convolution back to the main hdf5 file.

       Default value:
          True

       Validation:
          All(Range(min=None, max=1, min_included=True, max_included=True, msg=None), <function Boolean at 0x7f7d14f133a0>, msg=None)
   * - yield_rate_unit
     - Description:
          Unit used for the yield-rate data. NOTE: currently it is not possible to override this thoruh the data_dict column or layer entries.

       Default value:
          1.0 1 / solMass

       Validation:
          <function unit_validation at 0x7f7d14f13040>
