"""Export final tables and plots for the thesis results chapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRIC_ORDER = ["pesq", "stoi", "snr", "segsnr", "lsd"]
METHOD_ORDER = ["Noisy", "Spectral_Subtraction", "Wiener_Filter", "MaskNet"]
METHOD_LABELS = {
    "Noisy": "Noisy",
    "Spectral_Subtraction": "Spectral Subtraction",
    "Wiener_Filter": "Wiener Filter",
    "MaskNet": "MaskNet",
}
METRIC_LABELS = {
    "pesq": "PESQ",
    "stoi": "STOI",
    "snr": "SNR (dB)",
    "segsnr": "SegSNR (dB)",
    "lsd": "LSD",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export final thesis-ready result assets")
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        default=Path("results/evaluation"),
        help="Directory containing evaluation_stats.csv and improvements.csv",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/train_balanced_res_20260614_091134"),
        help="Checkpoint directory containing training_history.json and summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/final_report_assets"),
        help="Output directory for final tables and plots",
    )
    return parser.parse_args()


def load_inputs(evaluation_dir: Path, checkpoint_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], dict]:
    stats = pd.read_csv(evaluation_dir / "evaluation_stats.csv")
    improvements = pd.read_csv(evaluation_dir / "improvements.csv")
    with open(checkpoint_dir / "training_history.json", "r", encoding="utf-8") as handle:
        history = json.load(handle)
    with open(checkpoint_dir / "summary.json", "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    return stats, improvements, history, summary


def build_mean_table(stats: pd.DataFrame) -> pd.DataFrame:
    table = stats.pivot(index="method", columns="metric", values="mean")
    table = table.reindex(METHOD_ORDER)[METRIC_ORDER]
    table.index = [METHOD_LABELS[name] for name in table.index]
    table.columns = [METRIC_LABELS[name] for name in table.columns]
    return table


def build_improvement_table(improvements: pd.DataFrame) -> pd.DataFrame:
    table = improvements.pivot(index="method", columns="metric", values="improvement")
    table = table.reindex([method for method in METHOD_ORDER if method != "Noisy"])[METRIC_ORDER]
    table.index = [METHOD_LABELS[name] for name in table.index]
    table.columns = [METRIC_LABELS[name] for name in table.columns]
    return table


def latex_table(mean_table: pd.DataFrame, improvement_table: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Rezultatele medii obtinute pe setul de test VoiceBank-DEMAND. Pentru LSD, valori mai mici indica rezultate mai bune.}",
        r"\label{tab:rezultate_evaluare_finala}",
        r"\begin{tabular}{lrrrrr}",
        r"\hline",
        r"Metoda & PESQ & STOI & SNR (dB) & SegSNR (dB) & LSD \\",
        r"\hline",
    ]
    for method, row in mean_table.iterrows():
        lines.append(
            f"{method} & {row['PESQ']:.3f} & {row['STOI']:.3f} & "
            f"{row['SNR (dB)']:.3f} & {row['SegSNR (dB)']:.3f} & {row['LSD']:.3f} \\\\"
        )
    lines.extend(
        [
            r"\hline",
            r"\multicolumn{6}{l}{\textit{Imbunatatire fata de semnalul zgomotos}} \\",
        ]
    )
    for method, row in improvement_table.iterrows():
        lines.append(
            f"{method} & {row['PESQ']:+.3f} & {row['STOI']:+.3f} & "
            f"{row['SNR (dB)']:+.3f} & {row['SegSNR (dB)']:+.3f} & {row['LSD']:+.3f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def plot_metric_means(mean_table: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    axes = axes.flatten()
    colors = ["#8f8f8f", "#4c78a8", "#f58518", "#54a24b"]

    for idx, metric in enumerate(mean_table.columns):
        ax = axes[idx]
        values = mean_table[metric]
        ax.bar(values.index, values.values, color=colors, edgecolor="#333333", linewidth=0.6)
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=25)
        if metric == "LSD":
            ax.set_ylabel("Mai mic este mai bun")
        else:
            ax.set_ylabel("Mai mare este mai bun")

    axes[-1].axis("off")
    fig.suptitle("Comparatia metodelor de imbunatatire a vorbirii", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "final_metrics_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_improvements(improvement_table: pd.DataFrame, output_dir: Path) -> None:
    plot_table = improvement_table.copy()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    plot_table.plot(kind="bar", ax=ax, width=0.78)
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.set_title("Imbunatatiri fata de semnalul zgomotos", fontsize=13, weight="bold")
    ax.set_ylabel("Diferenta fata de Noisy")
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "final_improvements.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_training_history(history: list[dict], output_dir: Path) -> None:
    if not history:
        return
    frame = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(frame["epoch"], frame["train_loss"], marker="o", linewidth=2, label="Train loss")
    ax.plot(frame["epoch"], frame["val_loss"], marker="o", linewidth=2, label="Validation loss")
    ax.set_title("Evolutia functiei de pierdere in timpul antrenarii", fontsize=13, weight="bold")
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    ax.set_xticks(frame["epoch"])
    fig.tight_layout()
    fig.savefig(output_dir / "training_loss_curve.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary_text(
    mean_table: pd.DataFrame,
    improvement_table: pd.DataFrame,
    summary: dict,
    output_dir: Path,
) -> None:
    def to_markdown_table(frame: pd.DataFrame) -> str:
        rounded = frame.round(3)
        headers = ["Metoda", *rounded.columns]
        rows = [[idx, *[f"{value:.3f}" for value in row]] for idx, row in rounded.iterrows()]
        table_lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        table_lines.extend("| " + " | ".join(row) + " |" for row in rows)
        return "\n".join(table_lines)

    masknet_imp = improvement_table.loc["MaskNet"]
    lines = [
        "# Final Evaluation Summary",
        "",
        f"Experiment: `{summary.get('experiment_name', 'N/A')}`",
        f"Model: `{summary.get('model_variant', 'N/A')}`",
        f"Best validation loss: `{summary.get('best_val_loss', 'N/A')}`",
        "",
        "MaskNet obtine cele mai consistente imbunatatiri fata de semnalul zgomotos:",
        f"- PESQ: {masknet_imp['PESQ']:+.3f}",
        f"- STOI: {masknet_imp['STOI']:+.3f}",
        f"- SNR: {masknet_imp['SNR (dB)']:+.3f} dB",
        f"- SegSNR: {masknet_imp['SegSNR (dB)']:+.3f} dB",
        f"- LSD: {masknet_imp['LSD']:+.3f} (mai mic este mai bun)",
        "",
        "Tabel medii:",
        "",
        to_markdown_table(mean_table),
        "",
        "Tabel imbunatatiri:",
        "",
        to_markdown_table(improvement_table),
        "",
    ]
    (output_dir / "final_results_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stats, improvements, history, summary = load_inputs(args.evaluation_dir, args.checkpoint_dir)
    mean_table = build_mean_table(stats)
    improvement_table = build_improvement_table(improvements)

    mean_table.to_csv(args.output_dir / "final_metric_means.csv")
    improvement_table.to_csv(args.output_dir / "final_metric_improvements.csv")
    (args.output_dir / "final_results_table.tex").write_text(
        latex_table(mean_table, improvement_table),
        encoding="utf-8",
    )

    plot_metric_means(mean_table, args.output_dir)
    plot_improvements(improvement_table, args.output_dir)
    plot_training_history(history, args.output_dir)
    write_summary_text(mean_table, improvement_table, summary, args.output_dir)

    print(f"Final result assets saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
