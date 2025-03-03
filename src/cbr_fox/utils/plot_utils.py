import matplotlib.pyplot as plt
import numpy as np
def visualize_pyplot(cbr_fox_instance, **kwargs):
    """
    Visualize the best cases' components using Matplotlib.

    This method generates multiple plots to visualize the components of the best cases
    found by the Case-Based Reasoning (CBR) system. It visualizes the forecasted window,
    the prediction, and the best matching windows with the target windows. Each plot is
    customized based on parameters passed through `kwargs` for flexibility.

    Parameters
    ----------
    cbr_fox_instance : object
        An instance of the CBR system that contains the necessary data for plotting,
        such as the forecasted window, predictions, and best windows.

        kwargs : keyword arguments
            Additional arguments for customizing the plot appearance and behavior. The
            following options are supported:
            - 'forecast_label' : str, optional, default="Forecasted Window"
                The label for the forecasted window line in the plot.
            - 'prediction_label' : str, optional, default="Prediction"
                The label for the prediction point in the plot.
            - 'fmt' : str, optional
                The format string for plotting the best windows.
            - 'plot_params' : dict, optional
                Additional keyword arguments for customizing the best window plot.
            - 'scatter_params' : dict, optional
                Additional parameters for customizing the scatter plot for target windows.
            - 'xlim' : tuple, optional
                The limits for the x-axis (min, max).
            - 'ylim' : tuple, optional
                The limits for the y-axis (min, max).
            - 'xtick_rotation' : int, optional, default=0
                The rotation angle for x-axis tick labels.
            - 'xtick_ha' : str, optional, default='right'
                Horizontal alignment of the x-axis tick labels ('left', 'center', 'right').
            - 'title' : str, optional, default="Plot {i + 1}"
                The title for the plot.
            - 'xlabel' : str, optional, default="Axis X"
                The label for the x-axis.
            - 'ylabel' : str, optional, default="Axis Y"
                The label for the y-axis.
            - 'legend' : bool, optional, default=True
                Whether to display the legend in the plot.

    Returns
    -------
    list of tuples
        A list of tuples where each tuple contains a figure and axis object for
        each plot generated, which can be used for further customization or saving.

    Notes
    -----
    - The function will create a plot for each component in the target training
      windows based on the number of components available in the data.
    - This function requires a working instance of the CBR system, which holds the
      data for the best windows and predictions.
    """

    figs_axes = []
    n_windows = kwargs.get("n_windows", len(cbr_fox_instance.best_windows_index))
    # Un plot por cada componente
    for i in range(cbr_fox_instance.input_data_dictionary["target_training_windows"].shape[1]):
        fig, ax = plt.subplots()

        # Plot forecasted window and prediction
        ax.plot(
            np.arange(cbr_fox_instance.input_data_dictionary["window_len"]),
            cbr_fox_instance.input_data_dictionary["forecasted_window"][:, i],
            '--dk',
            label=kwargs.get("forecast_label", "Forecasted Window")
        )
        ax.scatter(
            cbr_fox_instance.input_data_dictionary["window_len"],
            cbr_fox_instance.input_data_dictionary["prediction"][i],
            marker='d',
            c='#000000',
            label=kwargs.get("prediction_label", "Prediction")
        )

        # Plot best windows
        for index in cbr_fox_instance.best_windows_index[:n_windows]:

            plot_args = [
                np.arange(cbr_fox_instance.input_data_dictionary["window_len"]),
                cbr_fox_instance.input_data_dictionary["training_windows"][index, :, i]
            ]
            if "fmt" in kwargs:
                plot_args.append(kwargs["fmt"])
            ax.plot(
                *plot_args,
                **kwargs.get("plot_params", {}),
                label=kwargs.get("windows_label", f"Window {index}")
            )
            ax.scatter(
                cbr_fox_instance.input_data_dictionary["window_len"],
                cbr_fox_instance.input_data_dictionary["target_training_windows"][index, i],
                **kwargs.get("scatter_params", {})
            )

        ax.set_xlim(kwargs.get("xlim"))
        ax.set_ylim(kwargs.get("ylim"))
        ax.set_xticks(np.arange(cbr_fox_instance.input_data_dictionary["window_len"]))
        plt.xticks(rotation=kwargs.get("xtick_rotation", 0), ha=kwargs.get("xtick_ha", 'right'))
        ax.set_title(kwargs.get("title", f"Plot {i + 1}"))
        ax.set_xlabel(kwargs.get("xlabel", "Axis X"))
        ax.set_ylabel(kwargs.get("ylabel", "Axis Y"))

        if kwargs.get("legend", True):
            ax.legend()

        figs_axes.append((fig, ax))
        fig.show()
    return figs_axes


def visualize_combined_pyplot(cbr_fox_instance, **kwargs):
    """
    Visualize the combined data and best cases' components using Matplotlib.

    This method generates plots that display the forecasted window, prediction, and
    a combined data representation for each component in the dataset. The function
    helps in visually analyzing how the CBR system's predictions align with the combined
    data and best matching cases. Users can customize the plot appearance and behavior
    through Matplotlib configurations passed via `kwargs`.

    Parameters
    ----------
    cbr_fox_instance : object
        An instance of the CBR system that contains the necessary data for plotting,
        including forecasted windows, predictions, and combined records.

        kwargs : keyword arguments
            Additional arguments for customizing the plot appearance and behavior. Supported options:
            - 'forecast_label' : str, optional, default="Forecasted Window"
                The label for the forecasted window line.
            - 'prediction_label' : str, optional, default="Prediction"
                The label for the prediction point.
            - 'combined_label' : str, optional, default="Combined Data"
                The label for the combined data plot.
            - 'combined_target_label' : str, optional, default="Combined Target"
                The label for the scatter points representing the combined target values.
            - 'xlim' : tuple, optional
                The limits for the x-axis (min, max).
            - 'ylim' : tuple, optional
                The limits for the y-axis (min, max).
            - 'xtick_rotation' : int, optional, default=0
                The rotation angle for x-axis tick labels.
            - 'xtick_ha' : str, optional, default='right'
                Horizontal alignment of the x-axis tick labels ('left', 'center', 'right').
            - 'title' : str, optional, default="Combined Plot {i + 1}"
                The title for the plot.
            - 'xlabel' : str, optional, default="Axis X"
                The label for the x-axis.
            - 'ylabel' : str, optional, default="Axis Y"
                The label for the y-axis.
            - 'legend' : bool, optional, default=True
                Whether to display the legend in the plot.

    Returns
    -------
    list of tuples
        A list of tuples where each tuple contains a figure and axis object for
        each plot generated, allowing further customization or saving.

    Notes
    -----
    - The function generates a plot for each component in the dataset based on the
      number of available target training windows.
    - It overlays forecasted data with combined records to facilitate a direct comparison.
    - This function requires a valid `cbr_fox_instance` containing precomputed records.
    """
    figs_axes = []

    # Un plot por cada componente
    for i in range(cbr_fox_instance.input_data_dictionary["target_training_windows"].shape[1]):
        fig, ax = plt.subplots()

        # Plot forecasted window and prediction
        ax.plot(
            np.arange(cbr_fox_instance.input_data_dictionary["window_len"]),
            cbr_fox_instance.input_data_dictionary["forecasted_window"][:, i],
            '--dk',
            label=kwargs.get("forecast_label", "Forecasted Window")
        )
        ax.scatter(
            cbr_fox_instance.input_data_dictionary["window_len"],
            cbr_fox_instance.input_data_dictionary["prediction"][i],
            marker='d',
            c='#000000',
            label=kwargs.get("prediction_label", "Prediction")
        )

        # Plot combined data
        ax.plot(
            np.arange(cbr_fox_instance.input_data_dictionary["window_len"]),
            cbr_fox_instance.records_array_combined[0][1][:, i],
            '-or',
            label=kwargs.get("combined_label", "Combined Data")
        )
        ax.scatter(
            cbr_fox_instance.input_data_dictionary["window_len"],
            cbr_fox_instance.records_array_combined[0][2][i],
            marker='o',
            c='red',
            label=kwargs.get("combined_target_label", "Combined Target")
        )

        ax.set_xlim(kwargs.get("xlim"))
        ax.set_ylim(kwargs.get("ylim"))
        ax.set_xticks(np.arange(cbr_fox_instance.input_data_dictionary["window_len"]))
        plt.xticks(rotation=kwargs.get("xtick_rotation", 0), ha=kwargs.get("xtick_ha", 'right'))
        ax.set_title(kwargs.get("title", f"Combined Plot {i + 1}"))
        ax.set_xlabel(kwargs.get("xlabel", "Axis X"))
        ax.set_ylabel(kwargs.get("ylabel", "Axis Y"))

        if kwargs.get("legend", True):
            ax.legend()

        figs_axes.append((fig, ax))
        fig.show()
    return figs_axes
