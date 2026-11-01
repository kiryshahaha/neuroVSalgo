import time
from typing import Optional
import numpy as np
from ortools.graph.python.linear_sum_assignment import SimpleLinearSumAssignment

from src.solvers.base import BaseSolver, SolverResult


class OrToolsSolver(BaseSolver):
    """
    Промышленный оптимизированный C++ солвер из Google OR-Tools.
    Основан на релаксациях потоков минимальной стоимости и аукционных методах (Auction Algorithm).

    Сложность: в среднем O(N^2 log N).
    Гарантия решения: 100% глобальный оптимум.
    Служит золотым стандартом (Ground Truth) в данном исследовании.
    """

    def __init__(self, scaling_factor: float = 1_000_000.0):
        """
        :param scaling_factor: Коэффициент масштабирования для перевода вещественных
                               стоимостей float в целочисленные int64 (требование C++ ядра OR-Tools).
        """
        super().__init__(name="Google_OR_Tools")
        self.scaling_factor = float(scaling_factor)

    def solve(self, cost_matrix: np.ndarray) -> SolverResult:
        if cost_matrix.ndim != 2 or cost_matrix.shape[0] != cost_matrix.shape[1]:
            raise ValueError(f"Ожидается квадратная 2D матрица, получена форма: {cost_matrix.shape}")

        n = cost_matrix.shape[0]

        # Подготовка векторизованных индексов дуг для C++ API
        # left: [0, 0, ..., 1, 1, ..., n-1, n-1]
        # right: [0, 1, ..., n-1, 0, 1, ..., n-1]
        left_nodes = np.repeat(np.arange(n, dtype=np.int32), n)
        right_nodes = np.tile(np.arange(n, dtype=np.int32), n)

        # Масштабирование матриц с вещественными весами в целочисленные 64-битные веса
        if np.issubdtype(cost_matrix.dtype, np.floating):
            scaled_costs = np.round(cost_matrix.flatten() * self.scaling_factor).astype(np.int64)
        else:
            scaled_costs = cost_matrix.flatten().astype(np.int64)

        # Инициализация и замер чистого времени выполнения C++ солвера
        solver = SimpleLinearSumAssignment()
        solver.add_arcs_with_cost(left_nodes, right_nodes, scaled_costs)

        t_start = time.perf_counter_ns()
        status = solver.solve()
        t_end = time.perf_counter_ns()

        execution_time_ms = (t_end - t_start) / 1_000_000.0

        if status != solver.OPTIMAL:
            status_map = {
                solver.INFEASIBLE: "INFEASIBLE",
                solver.POSSIBLE_OVERFLOW: "POSSIBLE_OVERFLOW",
            }
            status_str = status_map.get(status, f"UNKNOWN_STATUS_{status}")
            return SolverResult(
                assignment=np.array([], dtype=np.int32),
                total_cost=float("inf"),
                execution_time_ms=execution_time_ms,
                is_feasible=False,
                status_info=status_str,
            )

        # Извлечение назначения: right_mate(i) возвращает назначенную задачу для i-го исполнителя
        assignment = np.array([solver.right_mate(i) for i in range(n)], dtype=np.int32)

        # Точная суммарная стоимость по исходной вещественной матрице затрат
        row_indices = np.arange(n)
        total_cost = float(cost_matrix[row_indices, assignment].sum())
        is_feasible = self.validate_assignment(assignment, n)

        return SolverResult(
            assignment=assignment,
            total_cost=total_cost,
            execution_time_ms=execution_time_ms,
            is_feasible=is_feasible,
            status_info="OPTIMAL",
        )
