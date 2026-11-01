import time
from typing import Optional, List
import numpy as np
import torch

from src.solvers.base import BaseSolver, SolverResult
from src.models.assignment_net import NeuralAssignmentModel


class NeuralSolver(BaseSolver):
    """
    Обертка над нейросетевой моделью NeuralAssignmentModel, реализующая
    стандартный интерфейс BaseSolver для участия в бенчмарках.

    Поддерживает:
    - Автоматический выбор устройства (GPU CUDA или CPU).
    - Замер чистого времени инференса и декодирования.
    - Векторизованный параллельный batch-инференс solve_batch(matrices)
      с динамическим разделением на микробатчи для безопасного использования VRAM.
    """

    def __init__(
        self,
        model: Optional[NeuralAssignmentModel] = None,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        super().__init__(name="Neural_Sinkhorn")

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = NeuralAssignmentModel().to(self.device)
            if checkpoint_path is not None:
                state_dict = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(state_dict)

        self.model.eval()

    def solve(self, cost_matrix: np.ndarray) -> SolverResult:
        n = cost_matrix.shape[0]

        # Подготовка тензора на нужном устройстве
        cost_tensor = torch.from_numpy(cost_matrix.astype(np.float32)).to(self.device)

        # Синхронизация GPU перед замером времени
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        t_start = time.perf_counter_ns()
        with torch.no_grad():
            assignment, _ = self.model.predict(cost_tensor)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        t_end = time.perf_counter_ns()
        execution_time_ms = (t_end - t_start) / 1_000_000.0

        # Расчет суммарной стоимости по исходной матрице
        rows = np.arange(n)
        total_cost = float(cost_matrix[rows, assignment].sum())
        is_feasible = self.validate_assignment(assignment, n)

        return SolverResult(
            assignment=assignment,
            total_cost=total_cost,
            execution_time_ms=execution_time_ms,
            is_feasible=is_feasible,
            status_info="NEURAL_PREDICTION",
        )

    def solve_batch(self, cost_matrices: np.ndarray) -> List[SolverResult]:
        """
        Параллельный пакетный инференс всей пачки матриц на GPU/CPU.
        Для больших размерностей (N >= 500) автоматически дробит большой батч
        на микробатчи для предотвращения исчерпания видеопамяти (CUDA OOM).
        """
        B, N, _ = cost_matrices.shape

        # Динамический выбор размера микробатча исходя из размера N
        if N >= 1000:
            micro_batch_size = 4
        elif N >= 500:
            micro_batch_size = 8
        else:
            micro_batch_size = B

        all_assignments = []
        total_time_ms = 0.0

        for start_idx in range(0, B, micro_batch_size):
            end_idx = min(start_idx + micro_batch_size, B)
            sub_batch = cost_matrices[start_idx:end_idx]

            cost_tensor = torch.from_numpy(sub_batch.astype(np.float32)).to(self.device)

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            t_start = time.perf_counter_ns()
            with torch.no_grad():
                sub_assignments, _ = self.model.predict(cost_tensor)

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            t_end = time.perf_counter_ns()
            total_time_ms += (t_end - t_start) / 1_000_000.0
            all_assignments.append(sub_assignments)

        assignments = np.concatenate(all_assignments, axis=0)
        time_per_item_ms = total_time_ms / B

        results = []
        rows = np.arange(N)
        for b in range(B):
            assign = assignments[b]
            cost = float(cost_matrices[b][rows, assign].sum())
            feasible = self.validate_assignment(assign, N)
            results.append(
                SolverResult(
                    assignment=assign,
                    total_cost=cost,
                    execution_time_ms=time_per_item_ms,
                    is_feasible=feasible,
                    status_info="NEURAL_BATCH_PREDICTION",
                )
            )

        return results
