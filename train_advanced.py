"""
Скрипт обучения с поддержкой смешанных размерностей (Mixed-N Training):
Обучает модель на случайно меняющихся размерностях N из [20, 40, 60],
чтобы модель уверенно обобщалась на большие N в бенчмарке.
"""
import argparse
from pathlib import Path
import time
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.data_generator import get_generator
from src.solvers import OrToolsSolver
from src.models import NeuralAssignmentModel


def generate_batch_with_targets(batch_size: int, n: int, solver: OrToolsSolver, dist_types: list[str]):
    """
    Генерирует на лету батч матриц C и идеальные ответы OR-Tools X_true.
    """
    cost_batch = np.empty((batch_size, n, n), dtype=np.float32)
    target_batch = np.zeros((batch_size, n, n), dtype=np.float32)

    for i in range(batch_size):
        d_type = random.choice(dist_types)
        gen = get_generator(d_type)
        mat = gen.generate(n)
        res = solver.solve(mat)

        for w_idx, t_idx in enumerate(res.assignment):
            target_batch[i, w_idx, t_idx] = 1.0
        cost_batch[i] = mat

    return (
        torch.from_numpy(cost_batch),
        torch.from_numpy(target_batch),
    )


def train_mixed(
    epochs: int = 25,
    batches_per_epoch: int = 40,
    batch_size: int = 32,
    n_candidates: list[int] = [20, 35, 50, 70],
    val_size: int = 100,
    lr: float = 1e-3,
    save_path: str = "checkpoints/best_model.pt",
    seed: int = 42,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("ОБУЧЕНИЕ МОДЕРНИЗИРОВАННОЙ МОДЕЛИ (GraphEdgeMLP + FastGpuDecoder)")
    print(f"Устройство: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Размерности обучения (Mixed-N): {n_candidates}")
    print(f"Эпох: {epochs} | Батчей за эпоху: {batches_per_epoch} | Размер батча: {batch_size}")
    print("=" * 70)

    solver = OrToolsSolver()
    dist_types = ["spatial_2d", "uniform", "lognormal"]

    # Валидационный фиксированный набор (размер N=50)
    val_n = 50
    print(f"Подготовка фиксированного валидационного набора (N={val_n}, {val_size} шт.)...")
    val_costs, val_targets = generate_batch_with_targets(val_size, val_n, solver, dist_types)
    val_costs = val_costs.to(device)
    val_targets = val_targets.to(device)

    model = NeuralAssignmentModel(hidden_dim=64, sinkhorn_iters=25, tau=0.08).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    def loss_fn(pred_prob, target_perm):
        eps = 1e-8
        return -torch.sum(target_perm * torch.log(pred_prob + eps)) / pred_prob.shape[0]

    best_val_loss = float("inf")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_accum = 0.0

        for _ in range(batches_per_epoch):
            # Случайно выбираем N для текущего батча
            cur_n = random.choice(n_candidates)
            c_batch, t_batch = generate_batch_with_targets(batch_size, cur_n, solver, dist_types)
            c_batch = c_batch.to(device)
            t_batch = t_batch.to(device)

            optimizer.zero_grad()
            prob_batch = model(c_batch)
            loss = loss_fn(prob_batch, t_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            train_loss_accum += loss.item()

        avg_train_loss = train_loss_accum / batches_per_epoch
        scheduler.step()

        # Валидация
        model.eval()
        with torch.no_grad():
            val_probs = model(val_costs)
            val_loss = loss_fn(val_probs, val_targets).item()

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            mark = "(*)"
        else:
            mark = ""

        cur_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Эпоха [{epoch:02d}/{epochs:02d}] | "
            f"Train Loss: {avg_train_loss:6.2f} | "
            f"Val Loss (N={val_n}): {val_loss:6.2f} | "
            f"LR: {cur_lr:.6f} {mark}"
        )

    elapsed = time.time() - start_time
    print("=" * 70)
    print(f"Обучение завершено за {elapsed:.1f} сек. Лучший Val Loss: {best_val_loss:.2f}")
    print(f"Веса сохранены в: {save_path}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batches", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_path", type=str, default="checkpoints/best_model.pt")
    args = parser.parse_args()

    train_mixed(
        epochs=args.epochs,
        batches_per_epoch=args.batches,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()
