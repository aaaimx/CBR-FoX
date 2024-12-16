from cbr_fox import cbr_fox
import plotly.graph_objects as go
import numpy as np
import utils.plot_utils as plot_utils
class cbr_fox_builder:
    def __init__(self, techniques):
        # Store techniques as a dictionary, where the key is the technique name and the value is the cbr_fox object
        self.techniques_dict = dict()
        for item in techniques:
            if isinstance(item.metric, str) :
                self.techniques_dict[item.metric] = item
            else:
                self.techniques_dict[item.metric.__name__] = item
    
    def explain_all_techniques(self,  training_windows, target_training_windows, forecasted_window, prediction, num_cases):
        for name in self.techniques_dict:
            self.techniques_dict[name].explain(training_windows, target_training_windows, forecasted_window, prediction, num_cases)

    def fit(self, training_windows, target_training_windows, forecasted_window):
        for name in self.techniques_dict:
            self.techniques_dict[name].fit(training_windows, target_training_windows, forecasted_window)

    def predict(self, prediction, num_cases):
        for name in self.techniques_dict:
            self.techniques_dict[name].predict(prediction, num_cases)

    # Override __getitem__ to allow dictionary-like access
    def __getitem__(self, technique_name):
        # Return the corresponding cbr_fox object for the requested technique
        if technique_name in self.techniques_dict:
            return self.techniques_dict[technique_name]
        else:
            raise KeyError(f"Technique '{technique_name}' not found.")

    def visualize_pyplot(self, **kwargs):

        return [plot_utils.visualize_pyplot(self.techniques_dict[name], **kwargs) for name in self.techniques_dict]
