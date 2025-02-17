import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np

# Create a Dash app
app = dash.Dash(__name__)

# Example data: multiple techniques, each with several windows (time series)
techniques = {
    "Technique 1": {
        "Window 1": np.random.rand(15, 3),  # 15 timesteps, 3 features
        "Window 2": np.random.rand(15, 3),
        "Predicted Window": np.random.rand(15, 3)  # Predicted values
    },
    "Technique 2": {
        "Window 3": np.random.rand(15, 3),
        "Window 4": np.random.rand(15, 3),
        "Predicted Window": np.random.rand(15, 3)
    }
}

# Feature labels
features = ["Feature 1", "Feature 2", "Feature 3"]

# Create the layout with dropdowns for selecting techniques, windows, and features
app.layout = html.Div([
    html.H1("Time Series Analysis Dashboard"),

    # Dropdown to select techniques
    html.Label("Select Technique"),
    dcc.Dropdown(
        id='technique-dropdown',
        options=[{'label': k, 'value': k} for k in techniques.keys()],
        value='Technique 1'  # Default value
    ),

    # Dropdown to select the target window
    html.Label("Select Target Window"),
    dcc.Dropdown(
        id='target-window-dropdown',
        value='Window 1'  # Default value
    ),

    # Dropdown to select predicted window
    html.Label("Select Predicted Window"),
    dcc.Dropdown(
        id='predicted-window-dropdown',
        value='Predicted Window'  # Default value
    ),

    # Checkbox to select features for the target window
    html.Label("Select Features for Target Window"),
    dcc.Checklist(
        id='target-feature-checklist',
        options=[{'label': f, 'value': i} for i, f in enumerate(features)],
        value=[0],  # Default to the first feature
        inline=True
    ),

    # Checkbox to select features for the predicted window
    html.Label("Select Features for Predicted Window"),
    dcc.Checklist(
        id='predicted-feature-checklist',
        options=[{'label': f, 'value': i} for i, f in enumerate(features)],
        value=[0],  # Default to the first feature
        inline=True
    ),

    # Graph to display time series
    dcc.Graph(id='time-series-graph')
])

# Callback to update the target window dropdown based on selected technique
@app.callback(
    Output('target-window-dropdown', 'options'),
    Input('technique-dropdown', 'value')
)
def set_target_window_options(selected_technique):
    return [{'label': k, 'value': k} for k in techniques[selected_technique].keys() if k != "Predicted Window"]

# Callback to update the graph based on selected windows and features
@app.callback(
    Output('time-series-graph', 'figure'),
    [Input('technique-dropdown', 'value'),
     Input('target-window-dropdown', 'value'),
     Input('predicted-window-dropdown', 'value'),
     Input('target-feature-checklist', 'value'),
     Input('predicted-feature-checklist', 'value')]
)
def update_graph(selected_technique, selected_target_window, selected_predicted_window, selected_target_features, selected_predicted_features):
    target_window_data = techniques[selected_technique][selected_target_window]
    predicted_data = techniques[selected_technique][selected_predicted_window]
    timesteps = np.arange(target_window_data.shape[0])  # Predefined time steps

    # Create traces for each selected feature from the target window
    traces = []

    # Target window traces
    for feature_idx in selected_target_features:
        traces.append(go.Scatter(
            x=timesteps,
            y=target_window_data[:, feature_idx],
            mode='lines',
            name=f"{features[feature_idx]} (Target) - {selected_target_window}",
            line=dict(dash='solid')
        ))

    # Predicted window traces
    for feature_idx in selected_predicted_features:
        traces.append(go.Scatter(
            x=timesteps,
            y=predicted_data[:, feature_idx],
            mode='lines',
            name=f"{features[feature_idx]} (Predicted) - {selected_predicted_window}",
            line=dict(dash='dash')
        ))

    # Create figure
    return {
        'data': traces,
        'layout': go.Layout(
            title=f"Time Series for {selected_target_window} and {selected_predicted_window}",
            xaxis={'title': 'Time Steps'},
            yaxis={'title': 'Feature Value'},
            showlegend=True
        )
    }

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)