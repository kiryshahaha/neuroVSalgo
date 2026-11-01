"""
Скрипт генерации полного бенчмарк-датасета для научной работы.
Генерирует тестовые матрицы всех размерностей и распределений,
сохраняя их в каталог data/benchmark/.

Запуск:
    python generate_dataset.py
    python generate_dataset.py --samples 100 --quick
"""
import argparse
from pathlib import Path
import time
import numpy as np
from tqdm import tqdm

from src.data_generator import (
    UniformGenerator,
    LognormalGenerator,
    Spatial2DGenerator,
    save_dataset,
)


# Каноническая сетка размерностей N согласно ТЗ
BENCHMARK_SIZES = [10, 25, 50, 100, 250, 500, 1000]

# Быстрая сетка для быстрой отладки
QUICK_SIZES = [10, 25, 50, 100, 250]


def generate_benchmark_datasets(
    output_dir: str = "data/benchmark",
    num_samples: int = 50,
    sizes: list[int] = BENCHMARK_SIZES,
    base_seed: int = 42,
):
    """
    Генерирует и сохраняет на диск стандартизированные наборы данных
    для всех 3 распределений и всех размерностей N.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    distributions = {
        "uniform": UniformGenerator(low=0.0, high=1.0, seed=base_seed),
        "lognormal": LognormalGenerator(mean=0.0, sigma=0.75, seed=base_seed + 1),
        "spatial_2d": Spatial2DGenerator(metric="euclidean", seed=base_seed + 2),
    }

    print("=" * 70)
    print("ГЕНЕРАЦИЯ ПОЛНОГО БЕНЧМАРК-ДАТАСЕТА ДЛЯ ЭКСПЕРИМЕНТОВ")
    print(f"Каталог сохранения: {out_path.resolve()}")
    print(f"Количество матриц на каждую конфигурацию (num_samples): {num_samples}")
    print(f"Размерности N: {sizes}")
    print(f"Распределения: {list(distributions.keys())}")
    print("=" * 70)

    total_tasks = len(distributions) * len(sizes)
    start_total_time = time.time()

    with tqdm(total=total_tasks, desc="Генерация файлов") as pbar:
        for dist_name, generator in distributions.items():
            for n in sizes:
                pbar.set_postfix({"dist": dist_name, "N": n})

                # Пакетная генерация матриц (num_samples, n, n)
                matrices = generator.generate_batch(batch_size=num_samples, n=n)

                filename = f"{dist_name}_N{n}_samples{num_samples}.npz"
                filepath = out_path / filename

                metadata = {
                    "distribution": dist_name,
                    "n": n,
                    "num_samples": num_samples,
                    "seed": generator.seed,
                    "dtype": str(matrices.dtype),
                }

                save_dataset(filepath=filepath, matrices=matrices, metadata=metadata)
                pbar.update(1)

    elapsed = time.time() - start_total_time
    print("=" * 70)
    print(f"Генерация завершена за {elapsed:.2f} сек.")
    print(f"Всего создано файлов: {total_tasks}")
    print(f"Файлы готовы для подачи в OR-Tools, SciPy и нейросеть.")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Генератор бенчмарк-датасета задач о назначениях")
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Количество матриц каждого типа и размера для усреднения метрик (по умолчанию: 50)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Быстрый режим (N до 250, без тяжелого 500 и 1000 для быстрой проверки)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/benchmark",
        help="Папка для сохранения датасетов",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Базовый seed псевдослучайного генератора",
    )

    args = parser.parse_args()
    sizes = QUICK_SIZES if args.quick else BENCHMARK_SIZES

    generate_benchmark_datasets(
        output_dir=args.output,
        num_samples=args.samples,
        sizes=sizes,
        base_seed=args.seed,
    )


if __name__ == "__main__":
    main()
