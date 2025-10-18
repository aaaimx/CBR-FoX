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
    fig_size = kwargs.get("fig_size", (20, 12))
    plt.rcParams['figure.figsize'] = fig_size
    n_windows = kwargs.get("n_windows", len(cbr_fox_instance.best_windows_index))
    # One plot per component
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
        plt.show()
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
    fig_size = kwargs.get("fig_size", (8, 5))
    plt.rcParams['figure.figsize'] = fig_size
    # One plot per component
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
        plt.show()
    return figs_axes

def visualize_correlation_per_window(cbr_fox_instance, **kwargs):
    """
    Plot correlation values per window using Matplotlib.

    Produce a simple line plot of the CBR system's correlation values for each
    window to help assess how well the forecasted window matches training windows.

    Parameters
    ----------
    cbr_fox_instance : object
        CBR system instance containing an array-like attribute
        `correlation_per_window` with correlation values for each window.

    kwargs : keyword arguments
        Optional plotting parameters:
        - fig_size : tuple, optional
            Figure size in inches (width, height). Default used elsewhere is (20, 12).
        - correlation_label : str, optional
            Label for the primary correlation line.
        - fmt : str, optional
            Matplotlib format string to pass to the second plot invocation.
        - plot_params : dict, optional
            Additional keyword arguments forwarded to ax.plot for the second plot.
        - xlim, ylim : tuple, optional
            Axis limits passed to ax.set_xlim / ax.set_ylim.
        - xtick_rotation : int, optional
            Rotation angle for x-axis tick labels.
        - title, xlabel, ylabel : str, optional
            Axis and title text.
        - legend : bool, optional
            Whether to enable the legend.

    Returns
    -------
    tuple
        (fig, ax) Matplotlib figure and axis objects for further customization or saving.
    """
    fig, ax = plt.subplots()
    fig_size = kwargs.get("fig_size", (8, 5))
    fig.set_size_inches(fig_size)
    ax.plot(
        np.arange(len(cbr_fox_instance.correlation_per_window)),
        cbr_fox_instance.correlation_per_window,
        label=kwargs.get("correlation_label", "Correlation per Window")
    )
    plot_args = [
        np.arange(cbr_fox_instance.correlation_per_window.shape[0]),
        cbr_fox_instance.correlation_per_window
    ]
    if "fmt" in kwargs:
        plot_args.append(kwargs["fmt"])
    ax.plot(
        *plot_args,
        **kwargs.get("plot_params", {}),
        label=kwargs.get("label","Correlation per Window")
    )
    plt.show()
    return fig, ax


def visualize_smoothed_correlation(cbr_fox_instance, **kwargs):
    """
    Visualize a smoothed correlation series with annotated peak and valley points.
    Parameters
    ----------
    cbr_fox_instance : object
        An object that must provide the following attributes:
          - smoothed_correlation : 1D array-like (numpy array, list, pandas Series)
              Smoothed correlation values to plot.
          - peak_index : int or array-like of int
              Index or indices of peak point(s) in `smoothed_correlation` to mark.
          - valley_index : int or array-like of int
              Index or indices of valley point(s) in `smoothed_correlation` to mark.
    kwargs : dict, optional
        Optional plotting parameters (all keys are optional):
          - fig_size : tuple(float, float)
              Figure size in inches. Default: (20, 12).
          - smoothed_label : str
              Label used for the primary line plot of `smoothed_correlation`.
              Default: "Smoothed Correlation per Window".
          - fmt : str
              Matplotlib format string for the second line plot (e.g. 'r--').
          - plot_params : dict
              Additional keyword arguments forwarded to ax.plot for the second plot.
          - label : str
              Label for the second plot (default falls back to smoothed_label).
          - peak_params : dict
              Keyword arguments forwarded to ax.scatter when marking peak(s).
          - valley_params : dict
              Keyword arguments forwarded to ax.scatter when marking valley(s).
    Returns
    -------
    tuple
        (fig, ax) where `fig` is the matplotlib.figure.Figure and `ax` is the
        matplotlib.axes.Axes created by the function.
    Behavior
    --------
    - Creates a new matplotlib figure and axis, applies `fig_size` if provided,
      plots the smoothed correlation as a line, optionally re-plots it with a
      format string and additional plot parameters, and marks the peak and valley
      points using scatter.
    - Calls plt.show() before returning the (fig, ax) pair.
    - Expects that `smoothed_correlation` supports len() and indexing with
      `peak_index` / `valley_index`. If those indices are lists/arrays, multiple
      points are marked.
    Notes
    -----
    - This function assumes `matplotlib.pyplot` is available as plt and `numpy` as np
      in the module scope.
    - No validation is performed on the provided indices or arrays; callers should
      ensure indices are within bounds.
    """

    fig, ax = plt.subplots()
    fig_size = kwargs.get("fig_size", (20, 12))
    fig.set_size_inches(fig_size)
    ax.plot(
        np.arange(len(cbr_fox_instance.smoothed_correlation)),
        cbr_fox_instance.smoothed_correlation,
        label=kwargs.get("smoothed_label", "Smoothed Correlation per Window")
    )
    plot_args = [
        np.arange(cbr_fox_instance.smoothed_correlation.shape[0]),
        cbr_fox_instance.smoothed_correlation
    ]
    if "fmt" in kwargs:
        plot_args.append(kwargs["fmt"])
    ax.plot(
        *plot_args,
        **kwargs.get("plot_params", {}),
        label=kwargs.get("label","Smoothed Correlation per Window")
    )
    ax.scatter(
                cbr_fox_instance.peak_index,
                cbr_fox_instance.smoothed_correlation[cbr_fox_instance.peak_index],
                **kwargs.get("peak_params", {})
            )
    ax.scatter(
                cbr_fox_instance.valley_index,
                cbr_fox_instance.smoothed_correlation[cbr_fox_instance.valley_index],
                **kwargs.get("valley_params", {})
            )
    plt.show()
    return fig, ax