from typing import Optional, Tuple
import numpy as np
from src.data_generator.base import BaseDataGenerator


class Spatial2DGenerator(BaseDataGenerator):
    """
    Генератор матриц затрат на основе евклидовых расстояний между точками
    исполнителей (workers) и задач (tasks) на плоскости.

    Эмулирует физическую диспетчеризацию курьеров / сервисных инженеров в городской среде.
    В таких матрицах присутствует метрическая структура (неравенство треугольника, кластеризация),
    что принципиально отличает их от белого шума.
    """

    def __init__(
        self,
        bounds: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
        metric: str = "euclidean",
        seed: Optional[int] = None,
    ):
        """
        :param bounds: Границы области (x_min, x_max, y_min, y_max). По умолчанию единичный квадрат.
        :param metric: Тип метрики расстояния: 'euclidean' (L2) или 'manhattan' (L1, сетка улиц).
        :param seed: Зерно псевдослучайного генератора.
        """
        super().__init__(seed=seed)
        x_min, x_max, y_min, y_max = bounds
        if x_min >= x_max or y_min >= y_max:
            raise ValueError(f"Некорректные границы области: {bounds}")
        if metric not in ("euclidean", "manhattan"):
            raise ValueError(f"Неизвестная метрика: {metric}. Допустимы 'euclidean' или 'manhattan'.")

        self.bounds = bounds
        self.metric = metric

    def _sample_points(self, n: int) -> np.ndarray:
        """Сэмплирует n 2D-точек в заданных границах: форма (n, 2)."""
        x_min, x_max, y_min, y_max = self.bounds
        xs = self.rng.uniform(x_min, x_max, size=n)
        ys = self.rng.uniform(y_min, y_max, size=n)
        return np.column_stack((xs, ys))

    def _compute_distance_matrix(self, workers: np.ndarray, tasks: np.ndarray) -> np.ndarray:
        """
        Вычисляет попарные расстояния между исполнителями и задачами.
        workers: (n, 2)
        tasks:   (n, 2)
        Возвращает: матрицу (n, n)
        """
        # Векторизованная разность координат через broadcasting: (n, 1, 2) - (1, n, 2) -> (n, n, 2)
        diff = workers[:, np.newaxis, :] - tasks[np.newaxis, :, :]

        if self.metric == "euclidean":
            # sqrt((x_w - x_t)^2 + (y_w - y_t)^2)
            dist = np.sqrt(np.sum(diff ** 2, axis=-1))
        elif self.metric == "manhattan":
            # |x_w - x_t| + |y_w - y_t|
            dist = np.sum(np.abs(diff), axis=-1)
        else:
            raise ValueError(f"Неподдерживаемая метрика: {self.metric}")

        return dist.astype(np.float64)

    def generate(self, n: int) -> np.ndarray:
        if n <= 0:
            raise ValueError(f"Размерность n ({n}) должна быть положительным числом.")
        workers = self._sample_points(n)
        tasks = self._sample_points(n)
        return self._compute_distance_matrix(workers, tasks)

    def generate_with_coordinates(self, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Генерирует матрицу затрат и возвращает исходные координаты точек.
        Полезно для визуализации карты распределения на защите/в статье.

        :return: (cost_matrix (n, n), workers_coords (n, 2), tasks_coords (n, 2))
        """
        if n <= 0:
            raise ValueError(f"Размерность n ({n}) должна быть положительным числом.")
        workers = self._sample_points(n)
        tasks = self._sample_points(n)
        cost_matrix = self._compute_distance_matrix(workers, tasks)
        return cost_matrix, workers, tasks

    def generate_batch(self, batch_size: int, n: int) -> np.ndarray:
        if batch_size <= 0 or n <= 0:
            raise ValueError("batch_size и n должны быть положительными числами.")
        x_min, x_max, y_min, y_max = self.bounds

        workers_x = self.rng.uniform(x_min, x_max, size=(batch_size, n, 1))
        workers_y = self.rng.uniform(y_min, y_max, size=(batch_size, n, 1))
        tasks_x = self.rng.uniform(x_min, x_max, size=(batch_size, 1, n))
        tasks_y = self.rng.uniform(y_min, y_max, size=(batch_size, 1, n))

        dx = workers_x - tasks_x
        dy = workers_y - tasks_y

        if self.metric == "euclidean":
            dist = np.sqrt(dx ** 2 + dy ** 2)
        else:
            dist = np.abs(dx) + np.abs(dy)

        return dist.astype(np.float64)
