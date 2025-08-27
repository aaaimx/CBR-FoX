Installation Guide
==================

This section will guide you through installing **CBR-FoX** and its dependencies.

Installing CBR-FoX
-------------------

Install via pip (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To install the latest stable version from PyPI, use:

.. code-block:: sh

   pip install cbr-fox

Install from Source (Latest Development Version)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To install the latest development version from GitHub:

.. code-block:: sh

   git clone https://github.com/aaaimx/CBRFoX.git
   cd cbr-fox
   pip install .

Dependencies
------------

CBR-FoX requires the following libraries:

- **Python**: >=3.8
- **Core Dependencies**:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `sktime` *(for time-series analysis)*
  - `matplotlib` *(for plotting utilities)*

If some dependencies are missing, you can install them manually:

.. code-block:: sh

   pip install numpy pandas scikit-learn sktime matplotlib

Optional Dependencies
----------------------

- **Jupyter Notebook**: If you plan to run examples in notebooks, install:

.. code-block:: sh

   pip install jupyter

Verifying the Installation
--------------------------

After installation, verify that CBR-FoX is correctly installed by running:

.. code-block:: sh

   python -c "import cbr_fox; print(cbr-fox.__version__)"

If installed correctly, this should print the version number.

Troubleshooting Installation Issues
------------------------------------

### `ModuleNotFoundError: No module named 'cbr-fox'`

**Solution**: Ensure that the package is installed in the correct environment.

.. code-block:: sh

   pip list | grep cbr-fox

If missing, reinstall using:

.. code-block:: sh

   pip install cbr-fox

`ImportError: cannot import name 'CBRFox'`

**Solution**: Ensure the installation was successful by running:

.. code-block:: sh

   pip show cbr-fox

If the package is not listed, try reinstalling.

Compatibility Issues on Linux (Read the Docs)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If running on **Linux** (such as Read the Docs), ensure the correct TensorFlow version is installed:

.. code-block:: sh

   pip install tensorflow==2.17.0

If using **Intel versions** on Windows:

.. code-block:: sh

   pip install tensorflow-intel==2.17.0

For more help, check the **FAQ & Troubleshooting** section.

Now that you have CBR-FoX installed, proceed to the **User Guide** for usage examples.