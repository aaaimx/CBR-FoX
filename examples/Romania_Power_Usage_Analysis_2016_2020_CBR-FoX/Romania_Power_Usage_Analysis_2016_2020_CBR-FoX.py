import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from cbr_fox.core import cbr_fox
from cbr_fox.builder import cbr_fox_builder
from cbr_fox.custom_distance import cci_distance
import numpy as np

# Load the saved data
data = np.load(os.path.join(os.path.dirname(__file__),
                             "Romania_Power_Usage_Analysis_2016_2020_CBR-FoX.npz"))

# Retrieve each variable
training_windows = data['training_windows']
forecasted_window = data['forecasted_window']
target_training_windows = data['target_training_windows']
windowsLen = data['windowsLen'].item()  # Extract single value from array
componentsLen = data['componentsLen'].item()
windowLen = data['windowLen'].item()
prediction = data['prediction']

techniques = [
    cbr_fox(metric=cci_distance, kwargs={"punished_sum_factor":.5}),
    cbr_fox(metric=cci_distance, kwargs={"punished_sum_factor":.7})
]
p = cbr_fox_builder(techniques)
p.fit(training_windows = training_windows,target_training_windows = target_training_windows.reshape(-1,1), forecasted_window = forecasted_window)
p.predict(prediction = prediction,num_cases=5)
# p.plot_correlation()

p.visualize_pyplot(
    fmt = '--d',
    scatter_params={"s": 50},
    xtick_rotation=50,
    title="Power Usage Predictions in Romania (2016-2020)",
    xlabel="Time (Months)",
    ylabel="Power Usage (kWh)"
)
import matplotlib.pyplot as plt
plt.show()