import torch
import numpy as np


class FastGpuGreedyDecoder:
    """
    Высокопроизводительный декодер на PyTorch/GPU.

    Вместо медленной сортировки N^2 элементов в NumPy (которая для N=1000
    сортирует 1 000 000 чисел на CPU за 140+ мс), данный декодер:
    1. Работает полностью в памяти GPU (без копирования больших тензоров в CPU).
    2. Использует итеративный параллельный argmax с подавлением маской (-inf).
    3. Выполняет декодирование целого батча матриц (B, N, N) параллельно.
    """

    @staticmethod
    @torch.no_grad()
    def decode_tensor(prob_matrix: torch.Tensor) -> np.ndarray:
        """
        :param prob_matrix: Тензор вероятностей (B, N, N) или (N, N) на CUDA/CPU.
        :return: Массив целочисленных назначений:
                 - для 2D входа: 1D np.ndarray длины N
                 - для 3D входа: 2D np.ndarray формы (B, N)
        """
        is_2d = prob_matrix.ndim == 2
        if is_2d:
            prob_matrix = prob_matrix.unsqueeze(0)

        B, N, _ = prob_matrix.shape
        device = prob_matrix.device

        # Создаем рабочую копию на том же устройстве (GPU)
        probs = prob_matrix.clone()

        # Тензор для записи результатов: assignments[b, worker] = task
        assignments = torch.full((B, N), -1, dtype=torch.long, device=device)

        # Маски занятых исполнителей (строк) и задач (столбцов)
        row_mask = torch.zeros((B, N), dtype=torch.bool, device=device)
        col_mask = torch.zeros((B, N), dtype=torch.bool, device=device)

        # Выполняем N итераций быстрого параллельного выбора
        # На каждом шаге находим максимальный элемент по строкам
        for step in range(N):
            # Маскируем уже использованные строки и столбцы
            probs.masked_fill_(row_mask.unsqueeze(-1), -1e9)
            probs.masked_fill_(col_mask.unsqueeze(-2), -1e9)

            # Находим глобальный максимум в каждой матрице батча
            # Разворачиваем (B, N*N)
            flat_probs = probs.view(B, -1)
            max_val, flat_idx = torch.max(flat_probs, dim=-1)

            # Проверяем, есть ли матрицы, где все элементы уже замаскированы
            valid_batch = max_val > -1e8
            if not valid_batch.any():
                break

            best_rows = flat_idx // N
            best_cols = flat_idx % N

            # Записываем назначение
            batch_indices = torch.arange(B, device=device)
            assignments[batch_indices, best_rows] = best_cols

            # Обновляем маски
            row_mask[batch_indices, best_rows] = True
            col_mask[batch_indices, best_cols] = True

        # Если для каких-то ячеек остался -1 (крайний случай дегенерации), закрываем тривиально
        if (assignments == -1).any():
            for b in range(B):
                unassigned_rows = torch.where(assignments[b] == -1)[0]
                if len(unassigned_rows) > 0:
                    used_tasks = set(assignments[b][assignments[b] != -1].tolist())
                    all_tasks = set(range(N))
                    free_tasks = list(all_tasks - used_tasks)
                    for r, t in zip(unassigned_rows.tolist(), free_tasks):
                        assignments[b, r] = t

        result_np = assignments.cpu().numpy().astype(np.int32)
        return result_np[0] if is_2d else result_np
