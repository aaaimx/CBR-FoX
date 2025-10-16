Troubleshooting
===============

This section provides solutions to common issues encountered when using CBR-FoX.

Installation Issues
-------------------

ModuleNotFoundError: No module named 'cbr_fox'
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Python cannot find the cbr_fox module after installation.

**Solution**:

1. Verify the package is installed:

.. code-block:: bash

   pip list | grep CBR-FoX

2. If not listed, reinstall:

.. code-block:: bash

   pip install CBR-FoX

3. Ensure you're using the correct Python environment:

.. code-block:: bash

   which python
   python --version

ImportError: cannot import name 'cbr_fox'
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Import fails even after installation.

**Solution**:

1. Check if the package is correctly installed:

.. code-block:: bash

   pip show CBR-FoX

2. Try importing from the correct path:

.. code-block:: python

   from cbr_fox.core import cbr_fox
   from cbr_fox.builder import cbr_fox_builder

3. If still failing, reinstall in development mode:

.. code-block:: bash

   git clone https://github.com/aaaimx/CBR-FoX.git
   cd CBR-FoX
   pip install -e .


Dependency Conflicts
~~~~~~~~~~~~~~~~~~~~

**Problem**: Version conflicts with NumPy, SciPy, or other dependencies.

**Solution**:

1. Create a clean virtual environment:

.. code-block:: bash

   python -m venv cbr_fox_env
   source cbr_fox_env/bin/activate  # Linux/Mac
   cbr_fox_env\\Scripts\\activate    # Windows

2. Install with specific versions:

.. code-block:: bash

   pip install numpy==2.0.2 scipy==1.13.1
   pip install CBR-FoX


Runtime Errors
--------------

ValueError: shapes not aligned
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Array shape mismatch during fit or predict.

**Solution**:

Ensure your data has the correct shape:

.. code-block:: python

   # Correct shapes:
   training_windows.shape     # [n_windows, window_len, n_features]
   target_training_windows.shape  # [n_windows, n_features]
   forecasted_window.shape    # [window_len, n_features]
   prediction.shape           # [n_features]

Example fix:

.. code-block:: python

   # If target_training_windows is 1D
   if target_training_windows.ndim == 1:
       target_training_windows = target_training_windows.reshape(-1, 1)

   # If prediction is 1D with single feature
   if prediction.ndim == 0:
       prediction = np.array([prediction])


NaN values in correlation results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: ``correlation_per_window`` contains NaN values.

**Solution**:

1. Check for NaN in input data:

.. code-block:: python

   print(np.isnan(training_windows).any())
   print(np.isnan(forecasted_window).any())

2. Remove or impute NaN values:

.. code-block:: python

   training_windows = np.nan_to_num(training_windows, nan=0.0)
   forecasted_window = np.nan_to_num(forecasted_window, nan=0.0)

3. Verify data ranges (avoid extreme values):

.. code-block:: python

   print(f"Training range: {training_windows.min()} to {training_windows.max()}")


Memory Errors with Large Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: ``MemoryError`` when processing large time series.

**Solution**:

1. Reduce the number of windows:

.. code-block:: python

   # Use a subset of windows
   training_windows = training_windows[:1000]  # First 1000 windows
   target_training_windows = target_training_windows[:1000]

2. Process in batches:

.. code-block:: python

   batch_size = 500
   for i in range(0, len(training_windows), batch_size):
       batch = training_windows[i:i+batch_size]
       # Process batch

3. Use a smaller window length or fewer features


Metric-Specific Issues
----------------------

DTW is extremely slow
~~~~~~~~~~~~~~~~~~~~~

**Problem**: Dynamic Time Warping takes too long.

**Solution**:

1. Use DTW with window constraint:

.. code-block:: python

   cbr = cbr_fox(metric="dtw", kwargs={"window": 0.1})

2. Consider faster alternatives:

.. code-block:: python

   # Euclidean is much faster
   cbr = cbr_fox(metric="euclidean")

3. Reduce data dimensionality first


Custom metric not working
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Custom callable metric throws errors.

**Solution**:

Ensure your custom metric follows the correct signature:

.. code-block:: python

   def custom_metric(input_data_dictionary, **kwargs):
       """
       Custom distance metric.

       Parameters
       ----------
       input_data_dictionary : dict
           Contains 'training_windows', 'forecasted_window', etc.
       **kwargs : dict
           Additional parameters

       Returns
       -------
       np.ndarray
           Shape [n_windows, 1] with distance values
       """
       n_windows = input_data_dictionary['windows_len']
       # Your computation here
       distances = np.random.rand(n_windows, 1)  # Example
       return distances

Test it before using:

.. code-block:: python

   # Create test input
   test_input = {
       'training_windows': np.random.randn(10, 12, 2),
       'forecasted_window': np.random.randn(12, 2),
       'windows_len': 10,
       'components_len': 2
   }

   result = custom_metric(test_input)
   assert result.shape == (10, 1), f"Wrong shape: {result.shape}"


Visualization Problems
----------------------

Plots not showing
~~~~~~~~~~~~~~~~~

**Problem**: ``visualize_pyplot()`` doesn't display plots.

**Solution**:

1. In Jupyter notebooks, use:

.. code-block:: python

   %matplotlib inline
   builder.visualize_pyplot()

2. In scripts, add:

.. code-block:: python

   import matplotlib.pyplot as plt
   builder.visualize_pyplot()
   plt.show()  # Force display

3. Save plots instead:

.. code-block:: python

   figs = builder.visualize_pyplot()
   for i, (fig, ax) in enumerate(figs):
       fig.savefig(f'plot_{i}.png')


Empty or incomplete plots
~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Plots appear but show no data.

**Solution**:

1. Verify data was fitted and predicted:

.. code-block:: python

   print(f"Best windows: {len(cbr.best_windows_index)}")
   print(f"Analysis report: {cbr.analysisReport is not None}")

2. Check num_cases value:

.. code-block:: python

   # Ensure num_cases isn't too large
   max_cases = len(cbr.best_windows_index)
   cbr.predict(prediction, num_cases=min(5, max_cases))


Performance Issues
------------------

Slow execution with large datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: CBR-FoX takes too long to process data.

**Solutions**:

1. **Use faster metrics**:

.. code-block:: python

   # Fast: euclidean, squared
   # Medium: pearson
   # Slow: dtw, edr, erp

2. **Reduce smoothness iterations**:

.. code-block:: python

   cbr = cbr_fox(metric="euclidean", smoothness_factor=0.1)  # Less smoothing

3. **Parallel processing** (for multiple techniques):

.. code-block:: python

   from concurrent.futures import ProcessPoolExecutor

   def fit_technique(technique):
       technique.fit(train_w, target_w, forecast_w)
       return technique

   with ProcessPoolExecutor() as executor:
       results = executor.map(fit_technique, techniques)


High memory usage
~~~~~~~~~~~~~~~~~

**Problem**: Process uses too much RAM.

**Solution**:

.. code-block:: python

   # Clear intermediate results
   del cbr.concaveSegments
   del cbr.convexSegments
   import gc
   gc.collect()


Docker Issues
-------------

Container fails to start
~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Docker container exits immediately.

**Solution**:

1. Check if tests exist:

.. code-block:: bash

   docker run cbr_fox ls tests/

2. Run without tests:

.. code-block:: bash

   docker run cbr_fox python -c "from cbr_fox.core import cbr_fox; print('OK')"

3. Check logs:

.. code-block:: bash

   docker logs <container_id>


Image size too large
~~~~~~~~~~~~~~~~~~~~

**Problem**: Docker image is several GB.

**Solution**:

Use the slim Dockerfile:

.. code-block:: bash

   docker build -f Dockerfile -t cbr_fox:slim .
   docker images cbr_fox:slim


Platform-Specific Issues
-------------------------

Windows: DLL load failed
~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Import fails with DLL errors on Windows.

**Solution**:

1. Install Microsoft Visual C++ Redistributable
2. Use conda instead of pip:

.. code-block:: bash

   conda create -n cbr_fox python=3.11
   conda activate cbr_fox
   conda install -c conda-forge numpy scipy scikit-learn
   pip install CBR-FoX


macOS: SSL Certificate errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Cannot download packages due to SSL errors.

**Solution**:

.. code-block:: bash

   /Applications/Python\\ 3.11/Install\\ Certificates.command
   pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org CBR-FoX


Linux: Permission denied errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Cannot write to installation directory.

**Solution**:

.. code-block:: bash

   # Install for user only
   pip install --user CBR-FoX

   # Or use virtual environment
   python -m venv venv
   source venv/bin/activate
   pip install CBR-FoX


Getting Help
------------

If you encounter issues not covered here:

1. **Check GitHub Issues**: https://github.com/aaaimx/CBR-FoX/issues
2. **Ask in Discussions**: https://github.com/aaaimx/CBR-FoX/discussions
3. **Email Support**: jerryperezperez@hotmail.com

When reporting issues, please include:

- CBR-FoX version: ``pip show CBR-FoX``
- Python version: ``python --version``
- Operating system
- Full error traceback
- Minimal reproducible example

Common Error Messages
---------------------

Quick reference for error messages:

+--------------------------------------------+-----------------------------------+
| Error Message                              | Solution                          |
+============================================+===================================+
| ``IndexError: list index out of range``   | Check num_cases vs available data |
+--------------------------------------------+-----------------------------------+
| ``TypeError: 'NoneType' object``          | Run fit() before predict()        |
+--------------------------------------------+-----------------------------------+
| ``KeyError: 'correlation'``               | Run predict() before get_report() |
+--------------------------------------------+-----------------------------------+
| ``AttributeError: 'cbr_fox' has no...``   | Update to latest version          |
+--------------------------------------------+-----------------------------------+

For additional help, see the `API Documentation <modules.html>`_ or `Examples <examples.html>`_.