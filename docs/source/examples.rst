Examples
========

These are a set of examples to illustrate the library's usage and how to get the most out of it.

Bitcoin Prediction
------------------

1. Import Necessary Libraries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :caption: Example code

   from cbr_fox.builder.cbr_fox_builder import cbr_fox_builder
   from cbr_fox.core import cbr_fox
   from cbr_fox.custom_distance.cci_distance import cci_distance
   import numpy as np


2. Load the Saved Data
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :caption: Code snippet example to load data from a npz file

   # Load the saved data
   data = np.load("../examples/Bitcoin_prediction/Bitcoin_Prediction.npz")


3. Retrieve Variables from the Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :caption: Code snippet example to fecth data from a npz file

   # Retrieve each variable
   training_windows = data['training_windows']
   forecasted_window = data['forecasted_window']
   target_training_windows = data['target_training_windows']
   windowsLen = data['windowsLen'].item()  # Extract single value from array
   componentsLen = data['componentsLen'].item()
   windowLen = data['windowLen'].item()
   prediction = data['prediction']


4. Define CBR-FoX Techniques
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :caption: Code snippet example to define techniques to work with

   # Define the CBR-FoX techniques with custom distance metrics
   techniques = [
       cbr_fox.cbr_fox(metric=cci_distance, kwargs={"punishedSumFactor": 0.5})
       #cbr_fox.cbr_fox(metric="edr"),
       #cbr_fox.cbr_fox(metric="dtw"),
       #cbr_fox.cbr_fox(metric="twe")
   ]


5. Build and Train the CBR-FoX Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :caption: Code snippet example to build and train the model with the loaded data and the defined techniques

   # Initialize the CBR-FoX builder
   p = cbr_fox_builder(techniques)

   # Train the model with the provided data
   p.fit(training_windows = training_windows,
         target_training_windows = target_training_windows,
         forecasted_window = forecasted_window)

.. code-block:: text
   :caption: Technique execution output

   2025-08-27 12:56:19,487 - INFO - Analizando conjunto de datos
   2025-08-27 12:56:19,488 - INFO - Calculando Correlación
   Windows procesadas:   0%|          | 0/2774 [00:00<?, ?it/s]2025-08-27 12:56:19,493 - INFO - Aplicando Correlación de Pearson
   Windows procesadas:   0%|          | 0/2774 [00:00<?, ?it/s]
   2025-08-27 12:56:19,494 - INFO - Aplicando Correlación de Pearson
   Windows procesadas:   0%|          | 0/2774 [00:00<?, ?it/s]
   Windows procesadas: 100%|██████████| 2774/2774 [00:01<00:00, 2475.70it/s]
   2025-08-27 12:56:20,618 - INFO - Aplicando Correlación Euclidiana
   Windows procesadas: 100%|██████████| 2774/2774 [00:00<00:00, 12825.47it/s]
   2025-08-27 12:56:20,841 - INFO - Computando análisis de CBR
   2025-08-27 12:56:20,842 - INFO - Suavizando Correlación
   2025-08-27 12:56:21,049 - INFO - Extrayendo crestas y valles
   2025-08-27 12:56:21,050 - INFO - Recuperando segmentos convexos y cóncavos
   2025-08-27 12:56:21,051 - INFO - Recuperando índices originales de correlación
   Segmentos cóncavos: 100%|██████████| 5/5 [00:00<?, ?it/s]
   Segmentos convexos: 100%|██████████| 5/5 [00:00<00:00, 4992.03it/s]
   2025-08-27 12:56:21,055 - INFO - Análisis finalizado



6. Make Predictions
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :caption: Code snippet example to make predictions and generate explanations

   # Make predictions and generate explanations
   p.predict(prediction = prediction, num_cases=3)

.. code-block:: text
   :caption: Terminal output of the prediction process

   2025-08-27 12:56:23,598 - INFO - Generando reporte de análisis


7. Visualize Results
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :caption: Code snippet example to visualize the results

   # Visualize the predictions and results
   p.visualize_pyplot(
       fmt = '--d',
       scatter_params={"s": 50},
       xtick_rotation=50,
       title="nombre",
       xlabel="x",
       ylabel="y"
   )


.. image:: _static/bitcoin_image_1.png
   :align: center
   :alt: Output figure 1
