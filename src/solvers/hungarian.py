import time
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.solvers.base import BaseSolver, SolverResult


class ScipyHungarianSolver(BaseSolver):
    """
    Классический точный солвер на базе Венгерского алгоритма (алгоритм Куна-Манкреса / Джонкера-Волгенанта).
    Использует нативный C-бэкенд scipy.optimize.linear_sum_assignment.

    Асимптотическая сложность: O(N^3).
    Гарантия решения: 100% глобальный оптимум.
    """

    def __init__(self):
        super().__init__(name="SciPy_Hungarian")

    def solve(self, cost_matrix: np.ndarray) -> SolverResult:
        """
        Решает LSAP с использованием скомпилированного Си-ядра SciPy.
        """
        if cost_matrix.ndim != 2 or cost_matrix.shape[0] != cost_matrix.shape[1]:
            raise ValueError(f"Ожидается квадратная 2D матрица, получена форма: {cost_matrix.shape}")

        n = cost_matrix.shape[0]

        # Замер чистого времени выполнения C-функции с высоким разрешением
        t_start = time.perf_counter_ns()
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        t_end = time.perf_counter_ns()

        execution_time_ms = (t_end - t_start) / 1_000_000.0

        # Вектор назначений: assignment[i] = назначенная задача для i-го исполнителя
        assignment = np.empty(n, dtype=np.int32)
        assignment[row_ind] = col_ind

        total_cost = float(cost_matrix[row_ind, col_ind].sum())
        is_feasible = self.validate_assignment(assignment, n)

        return SolverResult(
            assignment=assignment,
            total_cost=total_cost,
            execution_time_ms=execution_time_ms,
            is_feasible=is_feasible,
            status_info="OPTIMAL",
        )
