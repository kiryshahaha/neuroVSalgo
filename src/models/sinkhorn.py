import torch
import torch.nn as nn


class SinkhornLayer(nn.Module):
    """
    Дифференцируемый слой Синхорна (Sinkhorn-Knopp Normalization).

    Принимает сырую матрицу логитов S размерности (B, N, N) и переводит ее
    в квази-дважды стохастическую матрицу P из многогранника Биркгофа B_N.
    В матрице P:
        - Все элементы P_{ij} > 0
        - Суммы по строкам близки к 1: sum_j P_{ij} ≈ 1
        - Суммы по столбцам близки к 1: sum_i P_{ij} ≈ 1

    Вычисления проводятся в логарифмическом пространстве (log-domain Sinkhorn)
    через LogSumExp для исключения переполнения (overflow) или деления на ноль.
    """

    def __init__(self, num_iters: int = 20, tau: float = 0.1):
        """
        :param num_iters: Число чередующихся нормализаций строк и столбцов (L).
        :param tau: Температура сглаживания (temperature). Меньше tau -> острее вероятности.
        """
        super().__init__()
        self.num_iters = num_iters
        self.tau = tau

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        :param logits: Тензор сырых скоров (B, N, N) или (N, N).
        :return: Квази-дважды стохастическая матрица P (B, N, N) или (N, N).
        """
        is_2d = logits.ndim == 2
        if is_2d:
            logits = logits.unsqueeze(0)

        # Масштабирование температурой
        log_P = logits / self.tau

        for _ in range(self.num_iters):
            # 1. Нормализация по строкам (последнее измерение dim=-1):
            # log_P_{ij} = log_P_{ij} - logsumexp_k(log_P_{ik})
            log_P = log_P - torch.logsumexp(log_P, dim=-1, keepdim=True)

            # 2. Нормализация по столбцам (предпоследнее измерение dim=-2):
            # log_P_{ij} = log_P_{ij} - logsumexp_k(log_P_{kj})
            log_P = log_P - torch.logsumexp(log_P, dim=-2, keepdim=True)

        P = torch.exp(log_P)

        if is_2d:
            P = P.squeeze(0)

        return P
