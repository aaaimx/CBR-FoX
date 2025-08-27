Features
========

CBR-Fox provides a variety of features to facilitate Case-Based Reasoning.

Core Features
-------------
- **Flexible Similarity Computation**: Uses Pearson correlation, Euclidean distance, and custom metrics.
- **Support for Time-Series Data**: Integrates with `sktime` to handle temporal similarity.
- **Built-in Visualization**: Provides plotting utilities for case comparison.
- **Batch Analysis**: Process multiple cases efficiently.

Custom Distance Functions
-------------------------
CBR-Fox allows users to **define custom similarity functions** for tailored comparisons.

Example:

.. code-block:: python

   from cbr_fox.custom_distance import cci_distance
   similarity = cci_distance(case1, case2)

Explore the **Usage Guide** to see practical applications of these features.
