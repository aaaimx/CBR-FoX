import numpy as np
from ..adapters import sktime_interface
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def cci_distance(input_data_dictionary, punishedSumFactor):
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
    punishedSumFactor : float
        A factor applied to the sum of the normalized correlation to adjust the
        final computed correlation.

    Returns
    -------
    numpy.ndarray
        A 2D array of shape (n_windows, 1) representing the normalized and scaled
        correlation per window.
    """

    logging.info("Aplicando Correlación de Pearson")
    pearsonCorrelation = sktime_interface.distance_sktime_interface(input_data_dictionary, sktime_interface.pearson)

    logging.info("Aplicando Correlación Euclidiana")
    euclideanDistance = sktime_interface.distance_sktime_interface(input_data_dictionary, "euclidean")
    normalizedEuclideanDistance = (euclideanDistance - np.amin(euclideanDistance, axis=0)) / (np.amax(euclideanDistance, axis=0)-np.amin(euclideanDistance, axis=0))

    normalizedCorrelation = (.5 + (pearsonCorrelation - 2 * normalizedEuclideanDistance + 1) / 4)

    # To overcome 1-d arrays

    correlationPerWindow = np.sum(((normalizedCorrelation + punishedSumFactor) ** 2), axis=1)
    if (correlationPerWindow.ndim == 1):
        correlationPerWindow = correlationPerWindow.reshape(-1, 1)
    # Applying scale
    correlationPerWindow = (correlationPerWindow - min(correlationPerWindow)) / (max(correlationPerWindow)-min(correlationPerWindow))
    return correlationPerWindow
