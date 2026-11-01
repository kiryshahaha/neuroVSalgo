from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class SolverResult:
    """
    Стандартизированный результат решения задачи о назначениях.

    Attributes:
        assignment: Вектор длины N, где assignment[i] = j означает,
                    что i-й исполнитель назначен на j-ю задачу.
        total_cost: Суммарная стоимость полученного назначения sum(C[i, assignment[i]]).
        execution_time_ms: Чистое время вычислений в миллисекундах.
        is_feasible: Флаг валидности (соответствие строгой биекции, отсутствие коллизий).
        status_info: Дополнительная информация от солвера (например, статус завершения).
    """
    assignment: np.ndarray
    total_cost: float
    execution_time_ms: float
    is_feasible: bool
    status_info: str = "SUCCESS"


class BaseSolver(ABC):
    """
    Базовый интерфейс для всех солверов (как классических, так и нейросетевых).
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def solve(self, cost_matrix: np.ndarray) -> SolverResult:
        """
        Решает задачу о назначениях для одной матрицы затрат.

        :param cost_matrix: Квадратная матрица затрат C размерности (N, N).
        :return: SolverResult со сформированным назначением и метаданными.
        """
        pass

    def solve_batch(self, cost_matrices: np.ndarray) -> List[SolverResult]:
        """
        Решает задачу для пакета (батча) матриц затрат (B, N, N).
        По умолчанию выполняет последовательный прогон. Для нейросетей переопределяется
        на параллельный батчевый инференс.

        :param cost_matrices: Массив матриц формы (B, N, N).
        :return: Список объектов SolverResult.
        """
        return [self.solve(matrix) for matrix in cost_matrices]

    @staticmethod
    def validate_assignment(assignment: np.ndarray, n: int) -> bool:
        """
        Проверяет, является ли полученный вектор взаимно-однозначным отображением (перестановкой)
        размером N (каждый исполнитель получил строго одну уникальную задачу).
        """
        if assignment is None or len(assignment) != n:
            return False
        # Проверяем, что множество значений совпадает с {0, 1, ..., n-1}
        unique_tasks = np.unique(assignment)
        return len(unique_tasks) == n and np.all(unique_tasks == np.arange(n))
