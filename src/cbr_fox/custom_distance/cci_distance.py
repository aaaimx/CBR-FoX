import numpy as np
from ..adapters import sktime_interface
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def cci_distance(input_data_dictionary, punished_sum_factor):
    """
    Compute a combined correlation and distance measure using Pearson correlation
    and Euclidean distance, with a normalization factor applied.

    This function first computes the Pearson correlation and the Euclidean distance
    between training windows and target windows using the `sktime_interface`. Then,
    it normalizes the Euclidean distance and combines both the correlation and
    distance measures into a final value. The result is further scaled and returned.

    Parameters
    ----------
    input_data_dictionary : dict
        A dictionary containing processed input data, including training windows,
        target training windows, and any other necessary components for distance
        calculations.
    punished_sum_factor : float
        A factor applied to the sum of the normalized correlation to adjust the
        final computed correlation.

    Returns
    -------
    numpy.ndarray
        A 2D array of shape (n_windows, 1) representing the normalized and scaled
        correlation per window.
    """

    logging.info("Applyiing Pearson Correlation")
    pearson_correlation = sktime_interface.distance_sktime_interface(input_data_dictionary, sktime_interface.pearson)

    logging.info("Applying Euclidean Distance")
    euclidean_distance = sktime_interface.distance_sktime_interface(input_data_dictionary, "euclidean")
    normalized_euclidean_distance = (euclidean_distance - np.amin(euclidean_distance, axis=0)) / (np.amax(euclidean_distance, axis=0)-np.amin(euclidean_distance, axis=0))

    normalized_correlation = (.5 + (pearson_correlation - 2 * normalized_euclidean_distance + 1) / 4)

    # To overcome 1-d arrays

    correlation_per_window = np.sum(((normalized_correlation + punished_sum_factor) ** 2), axis=1)
    if (correlation_per_window.ndim == 1):
        correlation_per_window = correlation_per_window.reshape(-1, 1)
    # Applying scale
    correlation_per_window = (correlation_per_window - min(correlation_per_window)) / (max(correlation_per_window)-min(correlation_per_window))
    return correlation_per_window
