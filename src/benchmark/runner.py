from typing import List, Dict
import numpy as np
from tqdm import tqdm

from src.solvers.base import BaseSolver
from src.benchmark.metrics import BenchmarkSummary, evaluate_run


class BenchmarkRunner:
    """
    Оркестратор бенчмарка: проводит калибровку, прогревочные прогоны (Warm-up)
    и контролирует сбор замеров времени и стоимости.
    """

    def __init__(self, solvers: List[BaseSolver], reference_solver_name: str = "Google_OR_Tools"):
        self.solvers = solvers
        self.reference_name = reference_solver_name

    def warmup(self, n: int = 30, warmup_runs: int = 5) -> None:
        """
        Прогревочные вызовы для инициализации CUDA-контекста и кэшей процессора.
        Исключает аномальные задержки при первых вызовах.
        """
        dummy_matrix = np.random.uniform(0.1, 1.0, size=(n, n)).astype(np.float64)
        for solver in self.solvers:
            for _ in range(warmup_runs):
                _ = solver.solve(dummy_matrix)

    def run_benchmark(
        self,
        matrices: np.ndarray,
        dist_name: str,
    ) -> List[BenchmarkSummary]:
        """
        Запускает все зарегистрированные солверы на переданном наборе матриц (num_samples, n, n).
        """
        num_samples, n, _ = matrices.shape

        # Сначала прогоняем эталонный солвер OR-Tools для получения Ground Truth
        ref_solver = next((s for s in self.solvers if s.name == self.reference_name), self.solvers[0])

        solver_data: Dict[str, Dict] = {
            s.name: {"times": [], "costs": [], "feasibles": []}
            for s in self.solvers
        }

        # Последовательный замер на каждой матрице
        for i in range(num_samples):
            mat = matrices[i]
            for solver in self.solvers:
                res = solver.solve(mat)
                solver_data[solver.name]["times"].append(res.execution_time_ms)
                solver_data[solver.name]["costs"].append(res.total_cost)
                solver_data[solver.name]["feasibles"].append(res.is_feasible)

        # Эталонные показатели (OR-Tools)
        ref_costs = solver_data[ref_solver.name]["costs"]
        ref_mean_time = float(np.mean(solver_data[ref_solver.name]["times"]))

        # Сведение в отчеты BenchmarkSummary
        summaries = []
        for solver in self.solvers:
            s_name = solver.name
            summary = evaluate_run(
                solver_name=s_name,
                n=n,
                dist_name=dist_name,
                times_ms=solver_data[s_name]["times"],
                costs=solver_data[s_name]["costs"],
                feasibles=solver_data[s_name]["feasibles"],
                reference_costs=ref_costs,
                reference_mean_time_ms=ref_mean_time,
            )
            summaries.append(summary)

        return summaries

    def run_batch_benchmark(
        self,
        matrices: np.ndarray,
        dist_name: str,
    ) -> List[BenchmarkSummary]:
        """
        Запуск бенчмарка в пакетном режиме (Batching), где весь тензор (B, N, N)
        подается на обработку целиком.
        """
        num_samples, n, _ = matrices.shape
        summaries = []

        # Замер последовательного решения через OR-Tools как базы
        ref_solver = next((s for s in self.solvers if s.name == self.reference_name), self.solvers[0])
        ref_results = ref_solver.solve_batch(matrices)
        ref_times = [r.execution_time_ms for r in ref_results]
        ref_costs = [r.total_cost for r in ref_results]
        ref_mean_time = float(np.mean(ref_times))

        for solver in self.solvers:
            results = solver.solve_batch(matrices)
            times = [r.execution_time_ms for r in results]
            costs = [r.total_cost for r in results]
            feasibles = [r.is_feasible for r in results]

            summary = evaluate_run(
                solver_name=solver.name,
                n=n,
                dist_name=dist_name,
                times_ms=times,
                costs=costs,
                feasibles=feasibles,
                reference_costs=ref_costs,
                reference_mean_time_ms=ref_mean_time,
            )
            summaries.append(summary)

        return summaries
