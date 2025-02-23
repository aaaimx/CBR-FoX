import numpy as np
import plotly.graph_objects as go

# Example: Create a NumPy array for the time series data
data = np.array([10, 12, 13, 15, 14, 16, 18, 17, 19, 20])
data2 = np.array([11, 11, 11, 11, 11, 11, 11, 17, 19, 20])
# If you have a corresponding time index (e.g., days)
time_index = np.arange(len(data))  # Create [0, 1, 2, ..., 9]

# Create a Plotly figure
fig = go.Figure()

# Add a time series line trace
fig.add_trace(go.Scatter(x=time_index, y=data, mode='lines+markers', name='Time Series'))
fig.add_trace(go.Scatter(x=time_index, y=data2, mode='lines+markers', name='Time Series 2'))
# Add titles and labels
fig.update_layout(
    title='Time Series Data',
    xaxis_title='Time (e.g., Days)',
    yaxis_title='Value',
    showlegend=True,

)

# Display the plot
fig.show()