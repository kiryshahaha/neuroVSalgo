from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Настройка стилей для публикации в научной работе
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.titlesize": 14,
    "lines.linewidth": 2,
    "lines.markersize": 6,
})


def plot_latency_vs_n(df: pd.DataFrame, save_path: str = "results/latency_vs_n.png") -> None:
    """
    Построение графика динамики времени решения (Latency, мс) в зависимости от размерности N.
    Использует логарифмическую шкалу Y для наглядности полиномиального vs параллельного роста.
    """
    plt.figure(figsize=(9, 5.5))

    solvers = df["solver_name"].unique()
    markers = ["o", "s", "^", "D"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for idx, solver in enumerate(solvers):
        sub_df = df[df["solver_name"] == solver].groupby("n")["mean_time_ms"].mean().reset_index()
        plt.plot(
            sub_df["n"],
            sub_df["mean_time_ms"],
            label=solver,
            marker=markers[idx % len(markers)],
            color=colors[idx % len(colors)],
        )

    plt.yscale("log")
    plt.xlabel("Размерность задачи N (число исполнителей и задач)")
    plt.ylabel("Время вычисления (мс, log-шкала)")
    plt.title("Сравнение времени решения: Классические солверы vs Нейросеть")
    plt.legend(frameon=True)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_optimality_gap_vs_n(df: pd.DataFrame, save_path: str = "results/optimality_gap_vs_n.png") -> None:
    """
    График потери оптимальности (Optimality Gap, %) нейросетевой модели при росте N.
    """
    plt.figure(figsize=(9, 5.5))

    # Фильтруем только нейросетевой солвер
    nn_df = df[df["solver_name"] == "Neural_Sinkhorn"]
    if nn_df.empty:
        plt.close()
        return

    sns.lineplot(
        data=nn_df,
        x="n",
        y="optimality_gap_percent",
        hue="distribution",
        marker="o",
        palette="tab10",
    )

    plt.xlabel("Размерность задачи N")
    plt.ylabel("Потеря оптимальности (Optimality Gap, %)")
    plt.title("Отклонение нейросетевого решения от математического оптимума OR-Tools")
    plt.legend(title="Распределение данных", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_batching_speedup(df_batch: pd.DataFrame, save_path: str = "results/batching_speedup.png") -> None:
    """
    График ускорения в пакетном режиме (Batch Throughput) на GPU.
    """
    plt.figure(figsize=(9, 5.5))

    sns.barplot(
        data=df_batch,
        x="n",
        y="speedup_vs_ortools",
        hue="solver_name",
        palette="viridis",
    )

    plt.axhline(1.0, color="red", linestyle="--", label="Базовый уровень (OR-Tools = 1.0x)")
    plt.xlabel("Размерность задачи N")
    plt.ylabel("Ускорение (Speedup vs OR-Tools, раз)")
    plt.title("Фактор ускорения при пакетной обработке (Batch Processing)")
    plt.legend(frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
