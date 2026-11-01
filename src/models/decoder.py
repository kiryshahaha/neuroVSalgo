import numpy as np
import torch


class GreedyDecoder:
    """
    Быстрый векторизованный жадный проекционный декодер (Greedy Rounding).

    Проецирует непрерывную матрицу вероятностей P из многогранника Биркгофа B_N
    в строго допустимую дискретную матрицу перестановок X из P_N (биекция).

    Алгоритм:
    1. Элементы матрицы P сортируются по убыванию уверенности модели.
    2. Жадный выбор: ячейка с максимальной вероятностью (i, j) фиксируется (assignment[i] = j),
       если исполнитель i и задача j еще не заняты.
    3. Оставшиеся редкие коллизии разрешаются линейным доназначением за O(N).
    """

    @staticmethod
    def decode_single(prob_matrix: np.ndarray | torch.Tensor) -> np.ndarray:
        """
        :param prob_matrix: Матрица вероятностей (N, N).
        :return: Вектор назначений assignment длины N (assignment[i] = j).
        """
        if isinstance(prob_matrix, torch.Tensor):
            prob_matrix = prob_matrix.detach().cpu().numpy()

        n = prob_matrix.shape[0]
        assignment = np.full(n, -1, dtype=np.int32)
        row_used = np.zeros(n, dtype=bool)
        col_used = np.zeros(n, dtype=bool)

        # Сортируем все N*N элементов по убыванию вероятности
        # argsort возвращает 1D индексы в развернутом массиве
        flat_indices = np.argsort(-prob_matrix.ravel())
        rows = flat_indices // n
        cols = flat_indices % n

        assigned_count = 0
        for r, c in zip(rows, cols):
            if not row_used[r] and not col_used[c]:
                assignment[r] = c
                row_used[r] = True
                col_used[c] = True
                assigned_count += 1
                if assigned_count == n:
                    break

        # Защита от редких коллизий: если кто-то остался не назначенным
        if assigned_count < n:
            free_rows = np.where(~row_used)[0]
            free_cols = np.where(~col_used)[0]
            for r, c in zip(free_rows, free_cols):
                assignment[r] = c

        return assignment

    @staticmethod
    def decode_batch(prob_matrices: np.ndarray | torch.Tensor) -> np.ndarray:
        """
        :param prob_matrices: Тензор вероятностей (B, N, N).
        :return: Массив назначений формы (B, N).
        """
        if isinstance(prob_matrices, torch.Tensor):
            prob_matrices = prob_matrices.detach().cpu().numpy()

        B, N, _ = prob_matrices.shape
        assignments = np.empty((B, N), dtype=np.int32)
        for b in range(B):
            assignments[b] = GreedyDecoder.decode_single(prob_matrices[b])
        return assignments
