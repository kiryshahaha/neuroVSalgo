from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class BaseDataGenerator(ABC):
    """
    Абстрактный базовый класс для генераторов матриц затрат (Cost Matrices).
    """

    def __init__(self, seed: Optional[int] = None):
        """
        :param seed: Зерно псевдослучайного генератора для воспроизводимости.
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def set_seed(self, seed: Optional[int]) -> None:
        """Переустановка зерна генератора."""
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def generate(self, n: int) -> np.ndarray:
        """
        Генерирует одну квадратную матрицу затрат размерности (n, n).

        :param n: Количество исполнителей и задач.
        :return: Матрица затрат np.ndarray формы (n, n) типа float64.
        """
        pass

    def generate_batch(self, batch_size: int, n: int) -> np.ndarray:
        """
        Генерирует пакет (батч) матриц затрат размерности (batch_size, n, n).

        :param batch_size: Количество генерируемых матриц.
        :param n: Размерность каждой задачи.
        :return: Тензор матриц np.ndarray формы (batch_size, n, n) типа float64.
        """
        matrices = [self.generate(n) for _ in range(batch_size)]
        return np.stack(matrices, axis=0)
