"""
Скрипт обучения нейросетевой модели NeuralAssignmentModel.
Обучение по принципу Knowledge Distillation (Учитель: Google OR-Tools).

Запуск:
    python train.py --epochs 25 --batch_size 64 --train_size 3000 --val_size 500 --n 40
"""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.data_generator import get_generator
from src.solvers import OrToolsSolver
from src.models import NeuralAssignmentModel


class AssignmentDataset(Dataset):
    """
    Датасет для задачи о назначениях.
    Генерирует матрицы затрат C и размечает истинные перестановки X_true с помощью Google OR-Tools.
    """

    def __init__(self, size: int, n: int, dist_types: list[str] = None, seed: int = 42):
        self.size = size
        self.n = n
        if dist_types is None:
            dist_types = ["spatial_2d", "uniform", "lognormal"]
        self.dist_types = dist_types

        print(f"Генерация и разметка датасета через OR-Tools (размер: {size}, N: {n})...")
        solver = OrToolsSolver()

        self.cost_matrices = np.empty((size, n, n), dtype=np.float32)
        self.target_permutations = np.empty((size, n, n), dtype=np.float32)

        gen_dict = {
            d: get_generator(d, seed=seed + i * 100)
            for i, d in enumerate(dist_types)
        }

        for i in tqdm(range(size), desc="Разметка матриц"):
            d_type = dist_types[i % len(dist_types)]
            gen = gen_dict[d_type]
            mat = gen.generate(n)

            # Решение точным солвером OR-Tools
            res = solver.solve(mat)

            # Формирование бинарной матрицы перестановок (Ground Truth)
            p_matrix = np.zeros((n, n), dtype=np.float32)
            for worker_idx, task_idx in enumerate(res.assignment):
                p_matrix[worker_idx, task_idx] = 1.0

            self.cost_matrices[i] = mat
            self.target_permutations[i] = p_matrix

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.cost_matrices[idx]),
            torch.from_numpy(self.target_permutations[idx]),
        )


def train(
    epochs: int = 25,
    batch_size: int = 64,
    train_size: int = 3000,
    val_size: int = 500,
    n: int = 40,
    lr: float = 1e-3,
    save_path: str = "checkpoints/best_model.pt",
    seed: int = 42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("ЗАПУСК ОБУЧЕНИЯ НЕЙРОСЕТЕВОГО МОДУЛЯ (Edge-MLP + Sinkhorn)")
    print(f"Устройство (Device): {device}")
    if device.type == "cuda":
        print(f"Имя видеокарты: {torch.cuda.get_device_name(0)}")
    print(f"Размер обучающей матрицы N: {n}")
    print(f"Обучающая выборка: {train_size} | Валидационная: {val_size}")
    print(f"Эпох: {epochs} | Размер батча: {batch_size} | LR: {lr}")
    print("=" * 70)

    # 1. Подготовка данных
    train_dataset = AssignmentDataset(size=train_size, n=n, seed=seed)
    val_dataset = AssignmentDataset(size=val_size, n=n, seed=seed + 999)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 2. Модель, оптимизатор, планировщик
    model = NeuralAssignmentModel(hidden_dim=64, sinkhorn_iters=25, tau=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    # Функция потерь: кросс-энтропия между матрицей Синхорна P и эталоном X_true
    def loss_fn(pred_prob, target_perm):
        # pred_prob: (B, N, N) в диапазоне (0, 1)
        # target_perm: (B, N, N) со значениями {0, 1}
        eps = 1e-8
        ce_loss = -torch.sum(target_perm * torch.log(pred_prob + eps)) / pred_prob.shape[0]
        return ce_loss

    best_val_loss = float("inf")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # 3. Цикл эпох обучения
    start_train_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0

        for cost_batch, target_batch in train_loader:
            cost_batch = cost_batch.to(device)
            target_batch = target_batch.to(device)

            optimizer.zero_grad()
            prob_batch = model(cost_batch)
            loss = loss_fn(prob_batch, target_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            total_train_loss += loss.item() * cost_batch.size(0)

        avg_train_loss = total_train_loss / train_size

        # Валидация
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for cost_batch, target_batch in val_loader:
                cost_batch = cost_batch.to(device)
                target_batch = target_batch.to(device)
                prob_batch = model(cost_batch)
                val_loss = loss_fn(prob_batch, target_batch)
                total_val_loss += val_loss.item() * cost_batch.size(0)

        avg_val_loss = total_val_loss / val_size
        scheduler.step(avg_val_loss)

        # Сохранение наилучших весов
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            best_mark = "(* Сохранена лучшая модель)"
        else:
            best_mark = ""

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Эпоха [{epoch:02d}/{epochs:02d}] | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"LR: {current_lr:.6f} {best_mark}"
        )

    elapsed_time = time.time() - start_train_time
    print("=" * 70)
    print(f"Обучение завершено за {elapsed_time:.1f} сек ({elapsed_time/60:.2f} мин).")
    print(f"Лучший Val Loss: {best_val_loss:.4f}")
    print(f"Веса сохранены в: {Path(save_path).resolve()}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Скрипт обучения модели NeuralAssignmentModel")
    parser.add_argument("--epochs", type=int, default=20, help="Количество эпох")
    parser.add_argument("--batch_size", type=int, default=64, help="Размер батча")
    parser.add_argument("--train_size", type=int, default=3000, help="Размер обучающей выборки")
    parser.add_argument("--val_size", type=int, default=500, help="Размер валидационной выборки")
    parser.add_argument("--n", type=int, default=40, help="Размерность матрицы N для обучения")
    parser.add_argument("--lr", type=float, default=1e-3, help="Скорость обучения (learning rate)")
    parser.add_argument("--save_path", type=str, default="checkpoints/best_model.pt", help="Путь сохранения весов")
    parser.add_argument("--seed", type=int, default=42, help="Seed воспроизводимости")

    args = parser.parse_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_size=args.train_size,
        val_size=args.val_size,
        n=args.n,
        lr=args.lr,
        save_path=args.save_path,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
