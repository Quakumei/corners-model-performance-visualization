import logging
import typing as tp

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


class CornersModelPerformancePlotter:
    """
    Class for plotting data of the corners detection model
    """

    def __init__(self, save_dir: str = "plots", darkgrid: bool = False):
        if darkgrid:
            sns.set_theme(style="darkgrid")
        self.save_dir = save_dir

    def save_fig(self, fig: plt.Figure, name: str) -> str:
        """
        Save figure to the given directory.

        Args:
            name (str): Name of the figure
            fig (plt.Figure): Figure to save
        Returns:
            str: Path to the saved figure
        """
        save_path = f"{self.save_dir}/{name}.png"
        fig.savefig(save_path)
        return save_path

    @staticmethod
    def plot_prediction_statistics(
        y_gt: tp.Iterable, y_pred: tp.Iterable
    ) -> tp.Tuple[plt.Figure, plt.Axes]:
        """
        Plot confusion matrices.
        """
        titles_options = [
            ("Confusion matrix, without normalization", None),
            ("Normalized confusion matrix", "true"),
        ]
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        suptitle = "CF matrix for GT and RB corners counts"
        fig.suptitle(suptitle)
        for i, (title, normalize) in enumerate(titles_options):
            cf_matrix = confusion_matrix(y_gt, y_pred, normalize=normalize)
            sns.heatmap(
                cf_matrix, annot=True, ax=axs[i], cmap="Blues", fmt=".2f"
            )
            axs[i].set_title(title)

        return fig, axs

    @staticmethod
    def deviation_statistics(
        mean: tp.Iterable, max: tp.Iterable, min: tp.Iterable, subtitle: str
    ) -> tp.Tuple[plt.Figure, plt.Axes]:
        """
        Plot deviation statistics
        """
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Deviation statistics" + f" ({subtitle})")
        plots = {
            "Mean": mean,
            "Max": max,
            "Min": min,
        }
        for ax, (title, data) in zip(axs, plots.items()):
            sns.histplot(data, ax=ax, kde=True, bins=20, stat="density")
            perc_95 = np.percentile(data, 95)
            perc_5 = np.percentile(data, 5)
            ax.set_title(
                f"{title} (5th percentile: {perc_5:.2f}) "
                f"(95th percentile: {perc_95:.2f})"
            )
            ax.set_ylabel("Count (density)")
            ax.set_xlabel("Deviation, deg")

        fig.tight_layout(pad=1.0)

        return fig, axs

    @staticmethod
    def plot_best_worst_tops(
        data: pd.DataFrame, by: str = "mean", n: int = 10
    ):
        """
        Plot best and worst predictions by the given metric.
        """
        fig, axs = plt.subplots(2, 1, figsize=(10, 15))
        fig.suptitle("Best and worst predictions")

        df = data.groupby("name").mean()
        df = df[by]

        plots = {
            f"Best {n} predictions by {by}": True,
            f"Worst {n} predictions by {by}": False,
        }

        for ax, (title, ascending) in zip(axs, plots.items()):
            top_n = df.sort_values(ascending=ascending).head(n)

            sns.barplot(x=top_n.index, y=top_n, ax=ax)
            ax.set_title(title)
            ax.set_ylabel(f"{by} Deviation, deg")
            ax.set_xlabel(None)
            ax.tick_params(axis="x", labelrotation=60)

        axs[-1].set_xlabel("Room name")

        # Add margin between plots
        fig.tight_layout(pad=3.0)

        return fig, axs

    def draw_plots(self, data: pd.DataFrame) -> tp.Iterable[str]:
        """
        Draw all plots for the given data.
        """
        plots = []
        # Plot confusion matrices for corner count
        logging.debug("Plotting confusion matrices")
        fig, axs = self.plot_prediction_statistics(
            data["gt_corners"], data["rb_corners"]
        )
        plots.append(self.save_fig(fig, "confusion_matrices"))

        # Plot deviation statistics
        logging.debug("Plotting deviation statistics")
        fig, axs = self.deviation_statistics(
            data["mean"], data["max"], data["min"], "all"
        )
        plots.append(self.save_fig(fig, "deviation_statistics_all"))
        fig, axs = self.deviation_statistics(
            data["floor_mean"], data["floor_max"], data["floor_min"], "floor"
        )
        plots.append(self.save_fig(fig, "deviation_statistics_floor"))
        fig, axs = self.deviation_statistics(
            data["ceiling_mean"],
            data["ceiling_max"],
            data["ceiling_min"],
            "ceiling",
        )
        plots.append(self.save_fig(fig, "deviation_statistics_mean"))

        # Plot best and worst predictions
        logging.debug("Plotting best and worst predictions")
        fig, axs = self.plot_best_worst_tops(data)
        plots.append(self.save_fig(fig, "best_worst_predictions"))

        return plots


@click.command()
@click.option(
    "--data-filename",
    default="data/deviation.json",
    help="Path to the data file",
)
@click.option(
    "--save-dir", default="plots", help="Directory to save the plots"
)
@click.option(
    "--darkgrid/--no-darkgrid", default=True, help="Enable dark grid in plots"
)
def main(data_filename, save_dir, darkgrid):
    df = pd.read_json(data_filename)
    plotter = CornersModelPerformancePlotter(
        save_dir=save_dir, darkgrid=darkgrid
    )
    plotter.draw_plots(df)
    plt.show()


if __name__ == "__main__":
    main()
