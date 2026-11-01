from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class BenchmarkSummary:
    """
    Сводные метрики по серии запусков солвера на наборе задач.
    """
    solver_name: str
    n: int
    distribution: str
    num_samples: int
    mean_time_ms: float
    std_time_ms: float
    p95_time_ms: float
    mean_cost: float
    optimality_gap_percent: float  # Отклонение от OR-Tools в %
    feasibility_rate_percent: float  # Доля корректных перестановок в %
    speedup_vs_ortools: float  # Ускорение относительно OR-Tools (T_ortools / T_solver)


def compute_optimality_gap(cost: float, opt_cost: float) -> float:
    """
    Вычисляет относительную потерю оптимальности (Optimality Gap, Delta) в процентах:
        Delta = ((Cost - Cost_opt) / Cost_opt) * 100%
    """
    if opt_cost <= 0:
        return 0.0
    gap = ((cost - opt_cost) / opt_cost) * 100.0
    return max(0.0, float(gap))


def evaluate_run(
    solver_name: str,
    n: int,
    dist_name: str,
    times_ms: List[float],
    costs: List[float],
    feasibles: List[bool],
    reference_costs: List[float],
    reference_mean_time_ms: float,
) -> BenchmarkSummary:
    """
    Агрегирует единичные замеры одного эксперимента в итоговую статистику.
    """
    times_arr = np.array(times_ms, dtype=np.float64)
    costs_arr = np.array(costs, dtype=np.float64)
    ref_costs_arr = np.array(reference_costs, dtype=np.float64)

    mean_time = float(np.mean(times_arr))
    std_time = float(np.std(times_arr))
    p95_time = float(np.percentile(times_arr, 95))
    mean_cost = float(np.mean(costs_arr))

    # Вычисление среднего Optimality Gap по всей серии
    gaps = [
        compute_optimality_gap(c, ref)
        for c, ref in zip(costs_arr, ref_costs_arr)
    ]
    mean_gap = float(np.mean(gaps))

    feasibility_rate = (sum(feasibles) / len(feasibles)) * 100.0

    speedup = reference_mean_time_ms / mean_time if mean_time > 0 else 1.0

    return BenchmarkSummary(
        solver_name=solver_name,
        n=n,
        distribution=dist_name,
        num_samples=len(times_ms),
        mean_time_ms=mean_time,
        std_time_ms=std_time,
        p95_time_ms=p95_time,
        mean_cost=mean_cost,
        optimality_gap_percent=mean_gap,
        feasibility_rate_percent=feasibility_rate,
        speedup_vs_ortools=speedup,
    )
