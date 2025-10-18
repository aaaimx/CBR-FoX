import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from cbr_fox.core import cbr_fox
from cbr_fox.builder import cbr_fox_builder
from cbr_fox.custom_distance import cci_distance
import numpy as np

# Load the saved data
data = np.load(os.path.join(os.path.dirname(__file__), "weather_forecasting.npz"))

# Retrieve each variable
training_windows = data['training_windows']
forecasted_window = data['forecasted_window']
target_training_windows = data['target_training_windows']
windowsLen = data['windowsLen'].item()  # Extract single value from array
componentsLen = data['componentsLen'].item()
windowLen = data['windowLen'].item()
prediction = data['prediction']

techniques = [
    cbr_fox(metric=cci_distance, kwargs={"punished_sum_factor": 0.5})
    #cbr_fox.cbr_fox(metric="edr"),
    #cbr_fox.cbr_fox(metric="dtw"),
    #cbr_fox.cbr_fox(metric="twe")
]
p = cbr_fox_builder(techniques)
p.fit(training_windows = training_windows,target_training_windows = target_training_windows, forecasted_window = forecasted_window)
p.predict(prediction = prediction,num_cases=3)
# p.plot_correlation()

p.visualize_pyplot(
    fmt = '--o',
    legend = True,
    scatter_params = {"s": 80, "alpha": 0.6, "edgecolors": "black"},
    xtick_rotation = 30,
    title="Weather Forecasting",
    xlabel="Slices",
    ylabel="Metric Value"
)
import matplotlib.pyplot as plt
plt.show()