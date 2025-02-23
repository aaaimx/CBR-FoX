import numpy as np
import logging
from sktime.distances import distance
from typing import Callable, Union
from tqdm import tqdm


def pearson(x, y):
    """
    Computes the Pearson correlation coefficient between two arrays.
    This function uses NumPy's `corrcoef` method to calculate the Pearson correlation matrix
    and extracts the correlation coefficient between `x` and `y`.

    """
    return np.corrcoef(x, y)[0][1]


def distance_sktime_interface(input_data_dictionary, metric, kwargs={}):
    """
    Interface for computing pairwise distances using a custom or built-in sktime metric.

    This function calculates the correlation or distance between two arrays for each component of each window.
    It allows the use of either sktime's built-in distance metrics or user-defined custom metrics,
    offering flexibility in the analysis. If the computation results in any NaN values, the function
    logs an error and raises an exception to prevent invalid further calculations.
    """
    result = np.array(
        [distance(input_data_dictionary["forecasted_window"][:, current_component],
                  input_data_dictionary["training_windows"][current_window, :, current_component],
                  metric, **kwargs)
         for current_window in tqdm(range(input_data_dictionary["windows_len"]), desc="Windows procesadas", position=0)
         for current_component in range(input_data_dictionary["components_len"])]
    ).reshape(-1, input_data_dictionary["components_len"])

    # Validating any NaN value. Whether there is any NaN value, the program stops
    if np.isnan(result).any():
        logging.error("The result of the metric computation is NaN.")
        raise ValueError("The computation returned NaN values, which is not valid for further calculations.")

    return result


def compute_distance_interface(input_data_dictionary,
                               metric: Union[str, Callable[[np.ndarray, np.ndarray], float]],
                               kwargs):
    """
    Interface for computing pairwise distances using sktime metrics or custom callable functions.

    This function attempts to calculate the distance or correlation between arrays using the specified
    `metric`. It first tries to execute the metric as a sktime-compatible distance function. If that fails,
    it falls back to using the `metric` as a user-defined callable function. Any errors during the process
    are logged, and an exception is raised if all attempts fail.

    Parameters
    ----------
    input_data_dictionary: dict
        The dictionary containing preprocessed input data.
    metric: str or callable
        A string or callable object used to compute the distance or correlation between two arrays.
    kwargs
        Additional keyword arguments to pass to the metric. Default is an empty dictionary.
    Returns
    -------
    ndarray
        A numpy array containing the computed distances or correlations for each component or window.

    Raises
    ------
    ValueError
        If both the sktime metric and the custom callable fail to compute the distance.
    TypeError
        If `metric` is not a string or callable.

    Notes
    -----
    - The sktime metric interface is tried first. If it fails, the function attempts to execute `metric` as a callable.
    - All errors are logged for debugging purposes.
    """
    correlation_per_window = np.array([])
    # correlation_per_window = metric(input_data_dictionary, **kwargs)
    try:
        # Attempt to use sktime's distance interface
        return distance_sktime_interface(input_data_dictionary, metric, kwargs)
    except Exception as sktime_error:
        logging.warning(f"Failed with sktime metric: {sktime_error}")

    # Fallback to custom callable
    try:
        if callable(metric):
            return metric(input_data_dictionary, **kwargs)
        else:
            raise TypeError(f"Metric must be callable or sktime-compatible, got: {type(metric).__name__}")
    except Exception as custom_error:
        logging.error("Custom callable execution failed", exc_info=True)
        raise
