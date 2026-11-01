import torch
import torch.nn as nn


class EdgeMLP(nn.Module):
    """
    Перестановочно-эквивариантный энкодер матрицы затрат C (Permutation Equivariant Encoder).

    Ключевое свойство:
    Модель не привязана к фиксированному размеру N.
    Для каждой ячейки (i, j) вычисляется вектор локально-глобального контекста:
        h_{ij} = [ C_{ij},  mean_row(i),  mean_col(j),  mean_all ]
    где:
        - C_{ij}: локальная стоимость назначения i-го исполнителя на j-ю задачу.
        - mean_row(i): средняя стоимость i-го исполнителя по всем задачам (насколько он в целом 'дорогой').
        - mean_col(j): средняя стоимость j-й задачи по всем исполнителям (насколько задача в целом 'сложная').
        - mean_all: глобальное среднее значение всей матрицы для нормировки масштаба.

    Вектор h_{ij} пропускается через компактный многослойный перцептрон (MLP).
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        # Входная размерность признака для каждого ребра: 4
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, cost_matrix: torch.Tensor) -> torch.Tensor:
        """
        :param cost_matrix: Тензор матриц затрат формы (B, N, N) или (N, N).
        :return: Тензор сырых логитов (score matrix) формы (B, N, N) или (N, N).
        """
        is_2d = cost_matrix.ndim == 2
        if is_2d:
            cost_matrix = cost_matrix.unsqueeze(0)  # (1, N, N)

        B, N, M = cost_matrix.shape
        assert N == M, f"Матрица должна быть квадратной, получено ({N}, {M})"

        # 1. Извлечение агрегированных статистик
        # Среднее по строкам: (B, N, 1) -> растягиваем до (B, N, N)
        row_mean = cost_matrix.mean(dim=-1, keepdim=True).expand(-1, -1, N)

        # Среднее по столбцам: (B, 1, N) -> растягиваем до (B, N, N)
        col_mean = cost_matrix.mean(dim=-2, keepdim=True).expand(-1, N, -1)

        # Глобальное среднее матрицы: (B, 1, 1) -> растягиваем до (B, N, N)
        global_mean = cost_matrix.mean(dim=(-2, -1), keepdim=True).expand(-1, N, N)

        # 2. Формирование 4-мерного тензора признаков: (B, N, N, 4)
        features = torch.stack(
            [cost_matrix, row_mean, col_mean, global_mean],
            dim=-1,
        )

        # 3. Применение MLP к каждому ребру (B, N, N, 4) -> (B, N, N, 1) -> (B, N, N)
        logits = self.net(features).squeeze(-1)

        if is_2d:
            logits = logits.squeeze(0)

        return logits
