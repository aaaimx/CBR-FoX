import matplotlib.pyplot as plt
import numpy as np
def visualize_correlation_per_window(self, plt_oject, input_data_dictionary):
    pass

def visualize_pyplot(cbr_fox_instance, **kwargs):
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
