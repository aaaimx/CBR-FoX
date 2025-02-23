import logging
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from scipy import signal
from statsmodels.nonparametric.smoothers_lowess import lowess
from adapters import sktime_interface
class prueba:
    """
    Core class to perform calculations and analysis at technique-level depth.

    This class is used to preprocess the provided input data for performing correlation and find the best cases. Its}
    functionality follows classic AI library guidelines and standards such as scikit-learn and keras.

    Parameters
    -------
    metric : str or callable, optional
        The metric to use for correlation (default is "dtw").
    smoothness_factor : float, optional
        The smoothness factor for preprocessing (default is 0.2).
    kwargs : dict, optional
        Additional keyword arguments for customization.

    Methods
    -------
    __init__(self, metric, smoothness_factor, kwargs)
        Initializes the cbr_fox class with specified parameters.
    """

    def __init__(self, metric: str or callable = "dtw", smoothness_factor: float = .2, kwargs: dict = {}):
        """
        Initializes the cbr_fox class with specified parameters.
        Parameters
        ----------
        metric
        smoothness_factor
        kwargs
        """
        # Variables for setting
        self.metric = metric
        self.smoothness_factor = smoothness_factor
        self.kwargs = kwargs
        # Variables for results
        self.smoothed_correlation = None
        self.analysisReport = None
        self.analysisReport_combined = None
        self.best_windows_index = list()
        self.worst_windows_index = list()
        self.bestMAE = list()
        self.worstMAE = list()
        # Private variables for easy access by private methods
        self.correlation_per_window = None
        self.input_data_dictionary = None
        self.records_array = None
        self.records_array_combined = None
        self.dtype = [('index', 'i4'),
                      ('window', 'O'),
                      ('target_window', 'O'),
                      ('correlation', 'f8'),
                      ('MAE', 'f8')]

