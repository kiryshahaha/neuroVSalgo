"""
Главная точка входа для проведения масштабного вычислительного эксперимента.
Сравнивает:
1. SciPy (C-реализация венгерского алгоритма).
2. Google OR-Tools (C++ аукционный алгоритм, Ground Truth).
3. Neural_Sinkhorn (Edge-MLP + Sinkhorn + Greedy Decoder на PyTorch CUDA).

Сохраняет:
- Таблицы результатов в results/benchmark_summary.csv
- Графики latency_vs_n.png, optimality_gap_vs_n.png, batching_speedup.png

Запуск:
    python run_benchmark.py
    python run_benchmark.py --samples 20 --quick
"""
import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd
from tabulate import tabulate

from src.data_generator import get_generator
from src.solvers import ScipyHungarianSolver, OrToolsSolver, NeuralSolver
from src.benchmark import BenchmarkRunner
from src.visualization import (
    plot_latency_vs_n,
    plot_optimality_gap_vs_n,
    plot_batching_speedup,
)


QUICK_SIZES = [10, 25, 50, 100, 250]
FULL_SIZES = [10, 25, 50, 100, 250, 500, 1000]
DISTRIBUTIONS = ["spatial_2d", "uniform", "lognormal"]


def run(
    sizes: list[int] = QUICK_SIZES,
    distributions: list[str] = DISTRIBUTIONS,
    num_samples: int = 20,
    checkpoint_path: str = "checkpoints/best_model.pt",
    output_dir: str = "results",
    seed: int = 42,
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ЗАПУСК КОМПЛЕКСНОГО НАУЧНОГО БЕНЧМАРКА (Классические vs Нейросеть)")
    print(f"Размерности N: {sizes}")
    print(f"Распределения: {distributions}")
    print(f"Число испытаний на точку: {num_samples}")
    print(f"Файл весов нейросети: {checkpoint_path}")
    print("=" * 80)

    # Инициализация солверов
    scipy_solver = ScipyHungarianSolver()
    ortools_solver = OrToolsSolver()
    neural_solver = NeuralSolver(checkpoint_path=checkpoint_path)

    solvers = [scipy_solver, ortools_solver, neural_solver]
    runner = BenchmarkRunner(solvers=solvers, reference_solver_name="Google_OR_Tools")

    print("\nПрогрев (Warm-up) компонентов...")
    runner.warmup(n=30, warmup_runs=5)
    print("Прогрев завершен.\n")

    single_records = []
    batch_records = []

    start_total_time = time.time()

    for dist_name in distributions:
        print(f"\n>>> Тестирование на распределении: [{dist_name.upper()}]")
        gen = get_generator(dist_name, seed=seed)

        for n in sizes:
            print(f"  Вычисление для N = {n:4d} (батч {num_samples} шт.)...", end="", flush=True)
            matrices = gen.generate_batch(batch_size=num_samples, n=n)

            # 1. Поточечный замер задержки (Single Instance Latency)
            single_summaries = runner.run_benchmark(matrices, dist_name=dist_name)
            for s in single_summaries:
                single_records.append(s.__dict__)

            # 2. Пакетный замер пропускной способности (Batch Throughput)
            batch_summaries = runner.run_batch_benchmark(matrices, dist_name=dist_name)
            for s in batch_summaries:
                batch_records.append(s.__dict__)

            print(" ГОТОВО")

    df_single = pd.DataFrame(single_records)
    df_batch = pd.DataFrame(batch_records)

    # Сохранение CSV
    single_csv_path = out_dir / "benchmark_single_latency.csv"
    batch_csv_path = out_dir / "benchmark_batch_throughput.csv"
    df_single.to_csv(single_csv_path, index=False)
    df_batch.to_csv(batch_csv_path, index=False)

    print("\n" + "=" * 80)
    print("ПОСТРОЕНИЕ ГРАФИКОВ...")
    plot_latency_vs_n(df_single, save_path=str(out_dir / "latency_vs_n.png"))
    plot_optimality_gap_vs_n(df_single, save_path=str(out_dir / "optimality_gap_vs_n.png"))
    plot_batching_speedup(df_batch, save_path=str(out_dir / "batching_speedup.png"))
    print("Графики сохранены в директорию results/.")

    # Формирование и вывод сводной таблицы в консоль
    print("\n" + "=" * 80)
    print("СВОДНАЯ СРАВНИТЕЛЬНАЯ ТАБЛИЦА (Усреднение по всем распределениям)")
    print("=" * 80)

    # Пивот по времени выполнения
    pivot_time = df_single.pivot_table(
        index=["n"],
        columns="solver_name",
        values="mean_time_ms",
        aggfunc="mean",
    )

    # Пивот по ошибке (Optimality Gap, %)
    pivot_gap = df_single.pivot_table(
        index=["n"],
        columns="solver_name",
        values="optimality_gap_percent",
        aggfunc="mean",
    )

    # Пивот по батч-ускорению
    pivot_speedup = df_batch.pivot_table(
        index=["n"],
        columns="solver_name",
        values="speedup_vs_ortools",
        aggfunc="mean",
    )

    summary_rows = []
    for n_val in sizes:
        t_scipy = pivot_time.loc[n_val, "SciPy_Hungarian"] if "SciPy_Hungarian" in pivot_time.columns else 0.0
        t_ortools = pivot_time.loc[n_val, "Google_OR_Tools"] if "Google_OR_Tools" in pivot_time.columns else 0.0
        t_neural = pivot_time.loc[n_val, "Neural_Sinkhorn"] if "Neural_Sinkhorn" in pivot_time.columns else 0.0

        gap_neural = pivot_gap.loc[n_val, "Neural_Sinkhorn"] if "Neural_Sinkhorn" in pivot_gap.columns else 0.0
        batch_speedup = pivot_speedup.loc[n_val, "Neural_Sinkhorn"] if "Neural_Sinkhorn" in pivot_speedup.columns else 0.0

        summary_rows.append([
            n_val,
            f"{t_ortools:.3f}",
            f"{t_scipy:.3f}",
            f"{t_neural:.3f}",
            f"{gap_neural:.2f}%",
            f"{batch_speedup:.2f}x",
        ])

    headers = [
        "N",
        "OR-Tools (мс)",
        "SciPy (мс)",
        "Нейросеть (мс)",
        "Optimality Gap (%)",
        "Батч-ускорение GPU",
    ]
    print(tabulate(summary_rows, headers=headers, tablefmt="github"))

    total_time = time.time() - start_total_time
    print("=" * 80)
    print(f"Эксперимент завершен за {total_time:.1f} сек.")
    print("Файлы для ВКР/статьи:")
    print(f"  1. {single_csv_path}")
    print(f"  2. {batch_csv_path}")
    print(f"  3. {out_dir / 'latency_vs_n.png'}")
    print(f"  4. {out_dir / 'optimality_gap_vs_n.png'}")
    print(f"  5. {out_dir / 'batching_speedup.png'}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Бенчмарк точных и нейросетевых солверов")
    parser.add_argument("--samples", type=int, default=20, help="Число матриц на точку (по умолчанию: 20)")
    parser.add_argument("--quick", action="store_true", help="Быстрый прогон (N до 250)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Путь к весам")
    parser.add_argument("--out", type=str, default="results", help="Папка результатов")

    args = parser.parse_args()
    sizes = QUICK_SIZES if args.quick else FULL_SIZES

    run(
        sizes=sizes,
        num_samples=args.samples,
        checkpoint_path=args.checkpoint,
        output_dir=args.out,
    )


if __name__ == "__main__":
    main()
