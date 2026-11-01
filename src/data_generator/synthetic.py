from typing import Optional
import numpy as np
from src.data_generator.base import BaseDataGenerator


class UniformGenerator(BaseDataGenerator):
    """
    Генератор матриц затрат с непрерывным равномерным распределением U(low, high).
    Используется как базовый синтетический стресс-тест без корреляций между элементами.
    """

    def __init__(self, low: float = 0.0, high: float = 1.0, seed: Optional[int] = None):
        super().__init__(seed=seed)
        if low >= high:
            raise ValueError(f"low ({low}) должно быть строго меньше high ({high}).")
        self.low = low
        self.high = high

    def generate(self, n: int) -> np.ndarray:
        if n <= 0:
            raise ValueError(f"Размерность n ({n}) должна быть положительным числом.")
        return self.rng.uniform(low=self.low, high=self.high, size=(n, n)).astype(np.float64)

    def generate_batch(self, batch_size: int, n: int) -> np.ndarray:
        if batch_size <= 0 or n <= 0:
            raise ValueError("batch_size и n должны быть положительными числами.")
        return self.rng.uniform(low=self.low, high=self.high, size=(batch_size, n, n)).astype(np.float64)


class LognormalGenerator(BaseDataGenerator):
    """
    Генератор матриц с логнормальным распределением LogNormal(mean, sigma).
    Моделирует распределение реальных трудозатрат в IT/проектах с 'длинным хвостом'
    (большинство задач типовые/быстрые, некоторые требуют аномально больших ресурсов).
    """

    def __init__(
        self,
        mean: float = 0.0,
        sigma: float = 0.75,
        normalize_max: bool = False,
        seed: Optional[int] = None,
    ):
        """
        :param mean: Среднее базового нормального распределения.
        :param sigma: Стандартное отклонение базового нормального распределения.
        :param normalize_max: Если True, матрица нормируется делением на свой максимум (диапазон (0, 1]).
        :param seed: Зерно псевдослучайного генератора.
        """
        super().__init__(seed=seed)
        if sigma <= 0:
            raise ValueError(f"sigma ({sigma}) должна быть строго положительной.")
        self.mean = mean
        self.sigma = sigma
        self.normalize_max = normalize_max

    def generate(self, n: int) -> np.ndarray:
        if n <= 0:
            raise ValueError(f"Размерность n ({n}) должна быть положительным числом.")
        matrix = self.rng.lognormal(mean=self.mean, sigma=self.sigma, size=(n, n)).astype(np.float64)
        if self.normalize_max:
            matrix /= np.max(matrix)
        return matrix

    def generate_batch(self, batch_size: int, n: int) -> np.ndarray:
        if batch_size <= 0 or n <= 0:
            raise ValueError("batch_size и n должны быть положительными числами.")
        matrices = self.rng.lognormal(
            mean=self.mean, sigma=self.sigma, size=(batch_size, n, n)
        ).astype(np.float64)
        if self.normalize_max:
            max_vals = np.max(matrices, axis=(-2, -1), keepdims=True)
            matrices /= max_vals
        return matrices
