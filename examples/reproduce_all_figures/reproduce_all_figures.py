import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from cbr_fox.core import cbr_fox
from cbr_fox.builder import cbr_fox_builder
from cbr_fox.custom_distance import cci_distance
import numpy as np

# Load the saved data
data = np.load(os.path.join(os.path.dirname(__file__), "..", "Weather_forecasting", "Weather_forecasting.npz"))

# Retrieve each variable
training_windows = data['training_windows']
forecasted_window = data['forecasted_window']
target_training_windows = data['target_training_windows']
windowsLen = data['windowsLen'].item()  # Extract single value from array
componentsLen = data['componentsLen'].item()
windowLen = data['windowLen'].item()
prediction = data['prediction']

# Define the CBR-FoX techniques with custom distance metrics
techniques = [
    cbr_fox(metric=cci_distance,smoothness_factor=.04 ,kwargs={"punished_sum_factor": .5})
]

# Initialize the CBR-FoX builder
p = cbr_fox_builder(techniques)

# Train the model with the provided data
p.fit(training_windows = training_windows,
      target_training_windows = target_training_windows,
        forecasted_window = forecasted_window)

# View correlations
p.visualize_pyplot(mode="correlation", fig_size=(10,5), plot_params={"color":'#1F77B4'})
p.visualize_pyplot(mode="smoothed", fig_size=(10,5), plot_params={"color":'#1F77B4'})

# Make predictions and generate explanations
p.predict(prediction = prediction,num_cases=3)

p.techniques_dict['cci_distance'].get_analysis_report()

# Visualize the predictions and results
p.visualize_pyplot(
    fmt = '--o',
    legend = True,
    n_windows = 3,
    fig_size = (8, 5),
    scatter_params = {"s": 80, "alpha": 0.6, "edgecolors": "black"},
    xtick_rotation = 30,
    title="Weather Forecasting using CBR-FoX",
    xlabel="Timestamps",
    ylabel="Similarity",
)

p.visualize_pyplot(
    fmt = '--o',
    legend = True,
    mode = "combined",
    n_windows = 3,
    fig_size = (8, 5),
    scatter_params = {"s": 80, "alpha": 0.6, "edgecolors": "black"},
    xtick_rotation = 30,
    title="Weather Forecasting using CBR-FoX",
    xlabel="Timestamps",
    ylabel="Similarity",
)

