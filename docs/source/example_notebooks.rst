Example notebooks
=================
We have a set of notebooks that explain and show the usage of the SSPC features. The notebooks are also stored in the `examples directory in the repository <https://gitlab.com/dhendriks/syntheticstellarpopconvolve/-/tree/master/examples>`_

.. raw:: html



    <style>
        .notebook-gallery {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            justify-content: center;
        }

        .notebook-item {
            width: 220px;
            text-align: center;
            font-family: sans-serif;
        }

        .notebook-item a {
            display: block;
            border: 2px solid #ccc;
            border-radius: 10px;
            overflow: hidden;
            transition: transform 0.3s ease-in-out, border-color 0.3s ease-in-out, box-shadow 0.3s ease-in-out;
        }

        .notebook-item img {
            width: 100%;
            display: block;
            transition: transform 0.3s ease-in-out;
        }

        .notebook-item a:hover img {
            transform: scale(1.1);
        }

        .notebook-item a:hover {
            border-color: #007BFF;
            box-shadow: 0 4px 15px rgba(0, 123, 255, 0.4);
        }

        .notebook-caption {
            margin-top: 8px;
            font-size: 14px;
            color: #444;
        }
    </style>



    <div class="notebook-gallery">

        <div class="notebook-item">
		    <a href="examples/notebook_example_GW_merger_rate_density.html">
		        <img src="_static/notebook_cover_images/GW_notebook_cover_image.png" alt="GW Merger Rate Density" width="200">
		    </a>
            <div class="notebook-caption">GW merger rate notebook</div>
        </div>

    </div>


.. toctree::
    :maxdepth: 2
    :caption: Contents:

    examples/notebook_convolution_tutorial.ipynb
    examples/notebook_convolution_star_formation_functions.ipynb
    examples/notebook_convolution_use_cases.ipynb
    examples/Background.ipynb

    examples/notebook_tutorial_persistent_data_and_previous_convolution_results.ipynb
    examples/notebook_tutorial_convolution_by_sampling.ipynb
    examples/notebook_tutorial_convolution_by_integration.ipynb
    examples/notebook_tutorial_convolution_on_the_fly.ipynb
    examples/notebook_tutorial_inflate_ensemble.ipynb
    examples/notebook_tutorial_convolve_binned_data.ipynb

    examples/notebook_example_bincodex.ipynb
    examples/notebook_example_GCE.ipynb
    examples/notebook_example_GW_merger_rate_density.ipynb
    examples/notebook_example_LISA_UCB.ipynb
    examples/notebook_example_orbit_integration_and_supernova_kicks.ipynb
    examples/notebook_example_GAIA_populations.ipynb
