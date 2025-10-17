import logging
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from scipy import signal
from statsmodels.nonparametric.smoothers_lowess import lowess
from ..adapters import sktime_interface
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class cbr_fox:
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
        self.analysis_report = None
        self.analysis_report_combined = None
        self.best_windows_index = list()
        self.worst_windows_index = list()
        self.best_mae = list()
        self.worst_mae = list()
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



    # FIRST PRIVATE LAYER
    def _smoothe_correlation(self):
        """
        Smooth the correlation using the lowess method from scipy, applying the specified smoothness factor.

        This method applies the Lowess (Locally Weighted Scatterplot Smoothing) technique to smooth the
        correlation values for further analysis, with the smoothness factor used to control the degree
        of smoothing.

        Returns
        -------
        numpy.ndarray
            A numpy array representing the smoothed correlation values, which can be used for further analysis.
        """
        return lowess(self.__correlation_per_window, np.arange(len(self.__correlation_per_window)),
                      self.smoothness_factor)[:, 1]

    def _identify_valleys_peaks_indexes(self):
        """
         Identify the indices of valleys and peaks in the smoothed correlation array.

         This method uses SciPy's `argrelextrema` function to locate local minima (valleys) and
         local maxima (peaks) for further analysis.

         Returns
         -------
         tuple of numpy.ndarray
             A tuple containing two numpy arrays:
             - The first array represents the indices of valleys (local minima).
             - The second array represents the indices of peaks (local maxima).
         """
        return signal.argrelextrema(self.smoothed_correlation, np.less)[0], \
            signal.argrelextrema(self.smoothed_correlation, np.greater)[0]

    def _retreive_concave_convex_segments(self, windows_len):
        """
        Extract concave and convex segments from the smoothed correlation array.

        This method splits the smoothed correlation data into concave and convex segments based on
        the identified valley and peak indices, storing the results in private attributes.

        Parameters
        ----------
        windows_len : int
            The length of the windows, corresponding to the number of data points in the correlation array.

        Returns
        -------
        None
            This method does not return a value but stores the calculated concave and convex segments
            in the private attributes `self.concaveSegments` and `self.convexSegments`, respectively.
        """
        self.concave_segments = np.split(
            np.transpose(np.array((np.arange(windows_len), self.smoothed_correlation))),
            self.valley_index)
        self.convex_segments = np.split(
            np.transpose(np.array((np.arange(windows_len), self.smoothed_correlation))),
            self.peak_index)

    def _retreive_original_indexes(self):
        """
        Retrieve original data point indexes from concave and convex segments.

        This method processes the concave and convex segments to extract the original indexes
        of data points from the correlation array. The indexes are stored in the private attributes
        `self.best_windows_index` (for concave segments) and `self.worst_windows_index` (for convex segments).

        Returns
        -------
        None
            The extracted indexes are stored in private attributes for further analysis.
        """
        for split in tqdm(self.concave_segments, desc="Segmentos cóncavos"):
            self.best_windows_index.append(int(split[np.where(split == max(split[:, 1]))[0][0], 0]))
        for split in tqdm(self.convex_segments, desc="Segmentos convexos"):
            self.worst_windows_index.append(int(split[np.where(split == min(split[:, 1]))[0][0], 0]))

    def calculate_analysis(self, indexes, input_data_dictionary):
        """
        Compute analysis results and store them in a record array for reporting and visualization.

        This method processes data for the specified indices, extracting relevant information such as
        training windows, target windows, correlation values, and Mean Absolute Error (MAE). The results
        are stored in a structured record array.

        Parameters
        ----------
        indexes : list of int
            A list of indices corresponding to valleys and peaks in the non-smoothed correlation array.
        input_data_dictionary : dict
            A dictionary containing preprocessed input data, including training and target windows.

        Returns
        -------
        numpy.recarray
            A structured record array containing the following fields:
            - Index: The index of the data point.
            - Training window: The array representing the training window.
            - Target window: The array representing the target window.
            - Correlation value: The correlation value from the non-smoothed correlation array.
            - MAE: The Mean Absolute Error (MAE) value comparing the target window and the prediction.
        """
        return np.array([(index,
                          input_data_dictionary["training_windows"][index],
                          input_data_dictionary["target_training_windows"][index],
                          self.correlation_per_window[index],
                          mean_absolute_error(input_data_dictionary["target_training_windows"][index],
                                              input_data_dictionary["prediction"].reshape(-1, 1)))
                         for index in indexes], dtype=self.dtype)

    def calculate_analysis_combined(self, input_data_dictionary, mode):
        # SECOND PRIVATE LAYER
        def weighted_average(values, weights):
            """
            Calculate the weighted average of a set of values based on the provided weights.

            Parameters
            ----------
            values : numpy.ndarray
                A 3D array of values with shape (n_windows, window_len, components_len),
                where `n_windows` is the number of windows, `window_len` is the length of each window,
                and `components_len` is the number of features per timestep.
            weights : numpy.ndarray or list
                A 1D array or list of weights corresponding to the values. The length of `weights` must
                match the first dimension of `values` (i.e., `n_windows`).

            Returns
            -------
            numpy.ndarray
                A 2D array of shape (window_len, components_len) representing the weighted average
                of the input values.
            """
            weights = np.array(weights)[:, np.newaxis, np.newaxis]
            return np.sum(values * weights, axis=0) / np.sum(weights)

        results = []
        for index, indices in enumerate([self.best_windows_index, self.worst_windows_index]):
            selected_cases = indices[:input_data_dictionary["num_cases"]]
            # Promedio simple
            if mode == "weighted":
                # Promedio ponderado
                average = weighted_average(input_data_dictionary["training_windows"][selected_cases],
                                                self.correlation_per_window[selected_cases])
            elif mode == "simple":
                average = np.mean(input_data_dictionary["training_windows"][selected_cases], axis=0)
            else:
                raise ValueError(f'Mode "{mode}" is not supported. Try: "simple" or "weighted".')

            target_average = np.mean(input_data_dictionary["target_training_windows"][selected_cases], axis=0)
            correlation_mean = np.mean(self.correlation_per_window[selected_cases])

            # se sustituye np.mean(input_data_dictionary["target_training_windows"][selected_cases], axis=0)
            mae = mean_absolute_error(target_average, input_data_dictionary["prediction"].reshape(-1, 1))

            results.append((-index, average, target_average, correlation_mean, mae))

        return np.array(results, dtype=self.dtype)

    def _preprocess_input_data(self, training_windows, target_training_windows, forecasted_window):
        """
        Gather basic data information from the input variables.

        This method processes the input data used for model training and forecasting,
        creating a dictionary of relevant metadata.

        Parameters
        ----------
        training_windows : numpy.ndarray
            A 3D array of training windows with shape (n_windows, window_len, components_len),
            where `n_windows` is the number of windows, `window_len` is the length of each window,
            and `components_len` is the number of features per timestep.
        target_training_windows : list
            A list of target windows corresponding to each training window.
        forecasted_window : list
            The window used for forecasting and making predictions.

        Returns
        -------
        dict
            A dictionary containing processed input data, including metadata such as
            `components_len`, `window_len`, and `windows_len`.
        """
        input_data_dictionary = dict()
        input_data_dictionary['training_windows'] = training_windows
        input_data_dictionary['target_training_windows'] = target_training_windows
        input_data_dictionary['forecasted_window'] = forecasted_window
        input_data_dictionary['components_len'] = training_windows.shape[2]
        input_data_dictionary['window_len'] = training_windows.shape[1]
        input_data_dictionary['windows_len'] = len(training_windows)

        return input_data_dictionary

    def _compute_correlation(self, input_data_dictionary):

        """
        Compute the correlation between training windows and target windows.

        This method uses the sktime interface or custom metrics to calculate
            correlations for further analysis and case retrieval.

        Parameters
        ----------
        input_data_dictionary : dict
            A dictionary containing processed input data, including training and
            target windows.

        Returns
        -------
        numpy.ndarray
            An array containing the correlation values for each window, normalized
            between 0 and 1.
        """

        # Implementing interface architecture to reduce tight coupling.
        correlation_per_window = sktime_interface.compute_distance_interface(input_data_dictionary, self.metric,
                                                                             self.kwargs)
        correlation_per_window = np.sum(correlation_per_window, axis=1)
        correlation_per_window = ((correlation_per_window - min(correlation_per_window)) /
                                  (max(correlation_per_window) - min(correlation_per_window)))
        self.correlation_per_window = correlation_per_window
        return correlation_per_window

    def _compute_cbr_analysis(self, input_data_dictionary):
        """
        Compute the analysis to smoothe, extract, retrieve indexes from the non-smoothed correlation

        This method smooths the window correlation, identifies peaks and valleys,
        extracts concave and convex segments, and retrieves the original indexes
        from the non-smoothed correlation array.

        Parameters
        ----------
        input_data_dictionary : dict
            A dictionary containing processed input data, which includes information
            about the window lengths.

        Returns
        -------
        None
            This method modifies internal attributes and does not return a value.
        """
        logging.info("Suavizando Correlación")
        self.smoothed_correlation = self._smoothe_correlation()
        logging.info("Extrayendo crestas y valles")
        self.valley_index, self.peak_index = self._identify_valleys_peaks_indexes()
        logging.info("Recuperando segmentos convexos y cóncavos")
        self._retreive_concave_convex_segments(input_data_dictionary['windows_len'])
        logging.info("Recuperando índices originales de correlación")
        self._retreive_original_indexes()

    def _compute_statistics(self, input_data_dictionary, mode):

        """
          Calculate statistics based on identified cases.

          This method performs calculations for both non-combined and combined results,
          selecting the appropriate case based on the number passed by the user.
          It generates a report for the user based on the computed statistics.

          Parameters
          ----------
          input_data_dictionary : dict
              A dictionary containing processed input data, including the number of cases to select.

          mode : str
              A string passed when the instance is created, used to set the strategy for combining cases.

          Returns
          -------
          None
              This method performs operations to generate and set the results, including reports for the user.
          """

        self.records_array = self.calculate_analysis(self.best_windows_index + self.worst_windows_index,
                                                     input_data_dictionary)
        self.records_array = np.sort(self.records_array, order="correlation")[::-1]

        # Selecting just the number of elements according to num_cases variable
        # The conditional is to avoid duplicity in case records_arrays's shape is not greater than the selected num_cases
        if (self.records_array.shape[0] > (input_data_dictionary["num_cases"] * 2)):
            self.records_array = np.concatenate(
                (self.records_array[:input_data_dictionary["num_cases"]], self.records_array[
                                                                          -input_data_dictionary[
                                                                              "num_cases"]:]))

        self.records_array_combined = self.calculate_analysis_combined(input_data_dictionary, mode)

        logging.info("Generando reporte de análisis")
        self.analysis_report = pd.DataFrame(data=pd.DataFrame.from_records(self.records_array))
        self.analysis_report_combined = pd.DataFrame(data=pd.DataFrame.from_records(self.records_array_combined))

    # PUBLIC METHODS. ALL THESE METHODS ARE PROVIDED FOR THE USER. Public layer

    def fit(self, training_windows: np.ndarray, target_training_windows: np.ndarray, forecasted_window: np.ndarray):

        """
        Perform correlation analysis and identify cases based on the provided data.

        This method analyzes the provided training windows, calculates the correlation using
        the selected metric, and performs CBR analysis to identify cases based on the data.

        Parameters
        ----------
        training_windows : numpy.ndarray
            A 3D array of training windows with shape (n_windows, window_len, components_len),
            where `n_windows` is the number of windows, `window_len` is the length of each window,
            and `components_len` is the number of features per timestep.

        target_training_windows : list
            A list of target windows corresponding to each training window.

        forecasted_window : list
            A window used for forecasting and making predictions.

        Returns
        -------
        None
            This method performs calculations for correlation analysis and case-based reasoning.
        """
        logging.info("Analizando conjunto de datos")
        self.input_data_dictionary = self._preprocess_input_data(training_windows, target_training_windows,
                                                                 forecasted_window)
        logging.info("Calculando Correlación")
        self.__correlation_per_window = self._compute_correlation(self.input_data_dictionary)
        logging.info("Computando análisis de CBR")
        self._compute_cbr_analysis(self.input_data_dictionary)
        logging.info("Análisis finalizado")

    def predict(self, prediction, num_cases: int, mode):
        """
        Perform analysis to identify the best cases based on the provided prediction.

        This method computes the statistics to identify the best cases compared to the provided
        prediction and returns results that can be accessed through corresponding methods.

        Parameters
        ----------
        prediction : numpy.ndarray
            The prediction generated by the AI model.

        num_cases : int
            The maximum number of cases to identify.

        mode : str, optional
            A string to specify the mode for the combined case option, which can be either 'simple' or 'combined'.
            The default is "simple".

        Returns
        -------
        None
            This method performs the analysis, and results can be accessed via the corresponding methods.
        """
        self.input_data_dictionary['prediction'] = prediction
        self.input_data_dictionary['num_cases'] = num_cases
        self._compute_statistics(self.input_data_dictionary, mode)

    def get_analysis_report(self):
        """
        Access the analysis report containing the best cases based on the analysis.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the best cases and their respective information based on the analysis.
        """
        return self.analysis_report

    def get_analysis_report_combined(self):
        """
        Access the combined analysis report containing the best cases based on the combined analysis.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the combined best cases and their respective information based on the analysis.
        """
        return self.analysis_report_combined
