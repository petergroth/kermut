import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import DictConfig
import matplotlib.colors as mcolors
import hydra


def lighten_color(hex_color, percent):
    """Lighten (increase HSL lightness) hex color."""
    rgb_color = mcolors.hex2color(hex_color)  # Convert hex to RGB
    lighter_rgb_color = [
        min(1.0, c + percent / 100.0) for c in rgb_color
    ]  # Adjust each component

    lighter_hex_color = mcolors.rgb2hex(lighter_rgb_color)  # Convert RGB to hex
    return lighter_hex_color


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="barplots",
)
def main(cfg: DictConfig) -> None:
    sns.set_style("darkgrid")

    # Rewrite above lists as dictionary
    methods_to_color = {
        "gp_kermutP": "#636EFA",
        "gp_kermutBH": lighten_color("#636EFA", 15),
        "gp_kermutBH_oh": lighten_color("#636EFA", 30),
        "gp_oh_seq_rbf": "#EF553B",
        "gp_oh_seq_lin": lighten_color("#EF553B", 15),
        "oh_seq": lighten_color("#EF553B", 30),
        "gp_oh_mut_rbf": "#00CC96",
        "gp_oh_mut_lin": lighten_color("#00CC96", 15),
        "oh_mut": lighten_color("#00CC96", 30),
        "mean_prediction": "#AB63FA",
    }

    metrics = ["mse", "spearman", "pearson"]
    metrics = ["test_" + metric for metric in metrics]
    datasets = cfg.datasets
    n_samples = cfg.n_samples
    save_fig = cfg.save_fig
    methods = methods_to_color.keys()
    # methods = cfg.methods
    colors = [methods_to_color[method] for method in methods]

    for dataset in datasets:
        for n_sample in n_samples:
            df = pd.DataFrame(
                columns=[
                    "seed",
                    "train_mse",
                    "test_mse",
                    "train_spearman",
                    "test_spearman",
                    "train_r2",
                    "test_r2",
                    "train_pearson",
                    "test_pearson",
                    "method",
                ]
            )

            for method in methods:
                df_method = pd.read_csv(
                    f"results/regression/{dataset}/{n_sample}_samples_{method}.tsv",
                    sep="\t",
                )
                if method == "mean_prediction":
                    df_method = df_method[["test_mse"]]
                df_method["method"] = method
                df = pd.concat((df, df_method), join="outer")

            df = df[["method", *metrics]]

            df_g = df.melt(
                id_vars=["method"],
                value_vars=metrics,
                var_name="metric",
                value_name="value",
            )
            # df_g = df_g.groupby(["method", "metric"], as_index=False).agg(["mean", "sem"])
            # df_g.columns = df_g.columns.droplevel(0)
            # df_g.columns = ["Method", "Metric", "Mean", "SEM"]
            # df_g = df_g.dropna()

            fig, ax = plt.subplots(2, 2, figsize=(7, 7))
            for i, metric in enumerate(metrics):
                ii = i + 1 if metric != "test_mse" else i
                ax_i = ax.flatten()[ii]
                df_method = df_g[df_g["metric"] == metric]

                sns.barplot(
                    data=df_method,
                    x="metric",
                    y="value",
                    ax=ax_i,
                    hue="method",
                    palette=colors,
                    hue_order=methods,
                    saturation=1,
                    errorbar="se",
                )
                ax_i.set_xlabel("")
                ax_i.set_ylabel("")
                if ii == 0:
                    handles, labels = ax_i.get_legend_handles_labels()
                    ax[0, 1].legend(
                        handles,
                        labels,
                        loc="center",
                        bbox_to_anchor=(0.5, 0.5),
                    )
                ax_i.get_legend().remove()

                if metric in ["test_spearman", "test_pearson", "test_r2"]:
                    ax_i.set_ylim([0, 1])

            ax[0, 1].set_axis_off()

            plt.suptitle(f"Regression results: {dataset} ({n_sample} samples)", size=16)
            plt.tight_layout()
            if save_fig:
                plt.savefig(
                    f"figures/regression_results_{n_sample}_samples_{dataset}.png"
                )
            plt.show()


if __name__ == "__main__":
    main()
