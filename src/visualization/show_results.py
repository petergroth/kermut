import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src import COLORS

if __name__ == "__main__":
    sns.set_style("dark")

    methods = ["gp_kermut", "gp_oh_rbf", "oh_seq", "oh_mut", "mean_prediction"]
    datasets = ["GFP", "BLAT_ECOLX", "PARD3_10"]
    n_samples = 100

    for dataset in datasets:
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
                f"results/regression/{dataset}/{n_samples}_samples_{method}.tsv",
                sep="\t",
            )
            if method == "mean_prediction":
                df_method = df_method[["test_mse"]]
            df_method["method"] = method
            df = pd.concat((df, df_method), join="outer")

        df = df[["method", "test_mse", "test_spearman", "test_r2", "test_pearson"]]
        df_g = df.melt(
            id_vars=["method"],
            value_vars=[
                "test_mse",
                "test_spearman",
                "test_r2",
                "test_pearson",
            ],
            var_name="metric",
            value_name="value",
        )
        # df_g = df_g.groupby(["method", "metric"], as_index=False).agg(["mean", "sem"])
        # df_g.columns = df_g.columns.droplevel(0)
        # df_g.columns = ["Method", "Metric", "Mean", "SEM"]
        # df_g = df_g.dropna()

        fig, ax = plt.subplots(2, 2, figsize=(7, 7))
        for i, metric in enumerate(
                ["test_mse", "test_r2", "test_pearson", "test_spearman"]
        ):
            ax_i = ax.flatten()[i]
            df_method = df_g[df_g["metric"] == metric]
            if dataset == "BLAT_ECOLX" and metric == "test_r2":
                df_method = df_method[df_method["method"] != "oh_mut"]

            sns.barplot(
                data=df_method,
                x="metric",
                y="value",
                ax=ax_i,
                hue="method",
                palette=COLORS,
                hue_order=methods,
                saturation=1,
                errorbar="se",
            )
            ax_i.set_xlabel("")
            ax_i.set_ylabel("")
            if i != 1:
                ax_i.get_legend().remove()

            if metric in ["test_spearman", "test_pearson", "test_r2"]:
                ax_i.set_ylim([0, 1])

        plt.suptitle(f"Regression results: {dataset} ({n_samples} samples)", size=16)
        plt.tight_layout()
        plt.savefig(f"figures/regression_results_{dataset}_{n_samples}_samples.png")
        plt.show()
