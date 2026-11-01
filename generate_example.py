"""
Пример генерации данных и демонстрация работы модуля data_generator.
Запуск:
    python generate_example.py
"""
import numpy as np
from src.data_generator import (
    UniformGenerator,
    LognormalGenerator,
    Spatial2DGenerator,
    get_generator,
    save_dataset,
    load_dataset,
)


def main():
    print("=" * 60)
    print("ДЕМОНСТРАЦИЯ РАБОТЫ МОДУЛЯ ГЕНЕРАЦИИ ДАННЫХ")
    print("=" * 60)

    # 1. Генерация матрицы с равномерным распределением
    print("\n1. Равномерное распределение (Uniform U[0, 1]):")
    uniform_gen = UniformGenerator(low=0.0, high=1.0, seed=42)
    matrix_uniform = uniform_gen.generate(n=4)
    print(f"Форма матрицы: {matrix_uniform.shape}")
    print("Пример матрицы затрат 4x4:")
    print(np.round(matrix_uniform, 3))

    # 2. Генерация матрицы с логнормальным распределением (IT-задачи)
    print("\n2. Логнормальное распределение (Lognormal, эмуляция трудозатрат):")
    lognormal_gen = LognormalGenerator(mean=0.0, sigma=0.75, seed=42)
    matrix_lognormal = lognormal_gen.generate(n=4)
    print("Пример матрицы затрат 4x4:")
    print(np.round(matrix_lognormal, 3))

    # 3. Генерация пространственных данных (координаты исполнителей и задач на карте)
    print("\n3. Пространственное распределение (2D Евклидово, координаты в городе):")
    spatial_gen = Spatial2DGenerator(metric="euclidean", seed=42)
    cost_matrix, workers, tasks = spatial_gen.generate_with_coordinates(n=3)
    print(f"Координаты 3 исполнителей (workers):\n{np.round(workers, 3)}")
    print(f"Координаты 3 задач (tasks):\n{np.round(tasks, 3)}")
    print(f"Итоговая матрица расстояний между ними:\n{np.round(cost_matrix, 3)}")

    # 4. Генерация батча через фабричный метод get_generator
    print("\n4. Пакетная генерация (Батч из 8 матриц размера 50x50):")
    gen = get_generator("uniform", seed=100)
    batch = gen.generate_batch(batch_size=8, n=50)
    print(f"Форма сгенерированного батча: {batch.shape} (batch_size, n, n)")

    # 5. Сохранение датасета в файл и его загрузка
    save_path = "data/sample_dataset.npz"
    print(f"\n5. Сохранение батча в файл {save_path}...")
    save_dataset(
        filepath=save_path,
        matrices=batch,
        metadata={"distribution": "uniform", "n": 50, "batch_size": 8, "seed": 100},
    )
    print("Файл успешно сохранен.")

    print(f"Загрузка из файла {save_path}...")
    loaded_matrices, metadata = load_dataset(save_path)
    print(f"Загружен батч формы: {loaded_matrices.shape}")
    print(f"Метаданные: {metadata}")
    print("\n" + "=" * 60)
    print("Все проверки успешно пройдены!")
    print("=" * 60)


if __name__ == "__main__":
    main()
