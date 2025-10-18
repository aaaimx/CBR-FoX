from ..utils import plot_utils

class cbr_fox_builder:
    """
    A class for managing multiple techniques used in case-based reasoning (CBR) with cbr_fox objects.

    This class allows the user to store different techniques, explain them, fit them to training data, and make predictions.
    It provides an interface for visualizing the results of each technique using `plot_utils`.
    """

    def __init__(self, techniques):
        """
        Initializes the cbr_fox_builder with a list of techniques.

        Parameters
        ----------
        techniques: list
            A list of techniques (objects) that contain a metric (string or callable) for CBR.
        """

        print(f"DEBUG: cbr_fox_builder.__init__ called")
        print(f"DEBUG: techniques parameter type: {type(techniques)}")
        print(f"DEBUG: techniques parameter value: {techniques}")
        print(f"DEBUG: techniques length: {len(techniques) if hasattr(techniques, '__len__') else 'No length'}")

        # Store techniques as a dictionary, where the key is the technique name and the value is the cbr_fox object
        self.techniques_dict = dict()

        print(f"DEBUG: Starting loop through techniques...")

        for i, item in enumerate(techniques):
            print(f"DEBUG: Processing item {i}: {item}")
            print(f"DEBUG: Item type: {type(item)}")

            # Check if item has metric attribute
            if hasattr(item, 'metric'):
                print(f"DEBUG: Item has 'metric' attribute")
                print(f"DEBUG: item.metric = {item.metric}")
                print(f"DEBUG: item.metric type = {type(item.metric)}")

                if isinstance(item.metric, str):
                    print(f"DEBUG: Metric is string: '{item.metric}'")
                    print(f"DEBUG: Adding to dict with key: '{item.metric}'")
                    self.techniques_dict[item.metric] = item
                    print(f"DEBUG: Successfully added string metric to dict")
                else:
                    print(f"DEBUG: Metric is not string, checking for __name__ attribute...")

                    if hasattr(item.metric, '__name__'):
                        print(f"DEBUG: item.metric.__name__ = {item.metric.__name__}")
                        key = item.metric.__name__
                        print(f"DEBUG: Adding to dict with key: '{key}'")
                        self.techniques_dict[key] = item
                        print(f"DEBUG: Successfully added callable metric to dict")
                    else:
                        print(f"DEBUG: ERROR - item.metric has no __name__ attribute!")
                        print(f"DEBUG: item.metric attributes: {dir(item.metric)}")
                        # Try to get a name anyway
                        try:
                            key = str(item.metric)
                            print(f"DEBUG: Using string representation as key: '{key}'")
                            self.techniques_dict[key] = item
                        except Exception as e:
                            print(f"DEBUG: Failed to create key from metric: {e}")
                            raise
            else:
                print(f"DEBUG: ERROR - Item has no 'metric' attribute!")
                print(f"DEBUG: Item attributes: {dir(item)}")
                raise AttributeError(f"Technique object {item} has no 'metric' attribute")

        print(f"DEBUG: Final techniques_dict keys: {list(self.techniques_dict.keys())}")
        print(f"DEBUG: cbr_fox_builder.__init__ completed successfully")

    def explain_all_techniques(self,  training_windows, target_training_windows, forecasted_window, prediction, num_cases):
        """
        Explains all techniques provided by the user.

        This method loops through each technique stored in `techniques_dict` and calls the `explain` method
        of each one to provide an explanation of the given case.

        Parameters
        ----------
        training_windows: ndarray
            The training windows for the CBR model.
        target_training_windows: ndarray
            The target training windows for the CBR model.
        forecasted_window: ndarray
            The forecasted window for the CBR model.
        prediction: ndarray
            The prediction made by the CBR model.
        num_cases: int
            The number of cases used in the explanation.
        """
        for name in self.techniques_dict:
            self.techniques_dict[name].explain(training_windows, target_training_windows, forecasted_window, prediction, num_cases)

    def fit(self, training_windows, target_training_windows, forecasted_window):
        """
        Fits all techniques to the provided training data.

        This method calls the `fit` method of each technique stored in `techniques_dict`
        to train them using the provided data.

        Parameters
        ----------
        training_windows: ndarray
            The training windows for the CBR model.
        target_training_windows: ndarray
            The target training windows for the CBR model.
        forecasted_window: ndarray
            The forecasted window for the CBR model.
        """
        for name in self.techniques_dict:
            self.techniques_dict[name].fit(training_windows, target_training_windows, forecasted_window)

    def predict(self, prediction, num_cases, mode="simple"):
        """
        Makes predictions using all the techniques stored in `techniques_dict`.

        This method calls the `predict` method of each technique, passing the provided prediction and number of cases.

        Parameters
        ----------
        prediction: ndarray
            The predicted values for the given cases.
        num_cases: int
            The number of cases to predict.
        """
        for name in self.techniques_dict:
            self.techniques_dict[name].predict(prediction, num_cases, mode)

    # Override __getitem__ to allow dictionary-like access
    def __getitem__(self, technique_name):
        """
        Allows dictionary-like access to retrieve a specific technique.

        This method returns the corresponding cbr_fox object for the requested technique name.

        Parameters
        ----------
        technique_name: str
            The name of the technique to retrieve.

        Returns
        -------
        cbr_fox object
            The technique associated with the provided name.

        Raises
        ------
        KeyError
            If the requested technique name does not exist.
        """
        # Return the corresponding cbr_fox object for the requested technique
        if technique_name in self.techniques_dict:
            return self.techniques_dict[technique_name]
        else:
            raise KeyError(f"Technique '{technique_name}' not found.")

    def visualize_pyplot(self, mode="individual", **kwargs):
        """
        Visualizes the techniques using `plot_utils.visualize_pyplot`.

        This method creates visualizations for all the techniques in `techniques_dict` using `plot_utils`.

        Parameters
        ----------
        **kwargs:
            Additional keyword arguments to pass to the visualization function.

        Returns
        -------
        list
            A list of visualizations for each technique.
        """

        if mode == "individual":
            return [plot_utils.visualize_pyplot(self.techniques_dict[name], **kwargs) for name in self.techniques_dict]
        elif mode == "combined":
            return [plot_utils.visualize_combined_pyplot(self.techniques_dict[name], **kwargs) for name in self.techniques_dict]
        elif mode == "correlation":
            return [plot_utils.visualize_correlation_per_window(self.techniques_dict[name], **kwargs) for name in self.techniques_dict]
        elif mode == "smoothed":
            return [plot_utils.visualize_smoothed_correlation(self.techniques_dict[name], **kwargs) for name in self.techniques_dict]
        else:
            print(f"Mode '{mode}' not supported. Use 'individual', 'combined', 'correlation' or 'smoothed'.")
            return []