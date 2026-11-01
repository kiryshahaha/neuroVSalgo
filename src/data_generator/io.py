"""
Модуль для сохранения и загрузки сгенерированных датасетов на диск.
Позволяет зафиксировать одинаковые наборы матриц для честного сравнения
всех алгоритмов без погрешностей разного сэмплирования.
"""
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np


def save_dataset(
    filepath: str | Path,
    matrices: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Сохраняет пакет матриц и метаданные в сжатый формат .npz.

    :param filepath: Путь к сохраняемому файлу (.npz).
    :param matrices: Тензор матриц затрат (batch_size, n, n).
    :param metadata: Дополнительные параметры (dist_type, seed, params и т.д.).
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_dict: Dict[str, Any] = {"matrices": matrices}
    if metadata:
        save_dict["metadata"] = np.array(metadata, dtype=object)
    np.savez_compressed(path, **save_dict)


def load_dataset(filepath: str | Path) -> tuple[np.ndarray, Optional[Dict[str, Any]]]:
    """
    Загружает ранее сохраненный датасет матриц и метаданные.

    :param filepath: Путь к .npz файлу.
    :return: (matrices (batch_size, n, n), metadata dict)
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Файл датасета не найден: {path}")

    data = np.load(path, allow_pickle=True)
    matrices = data["matrices"]
    metadata = None
    if "metadata" in data:
        metadata = data["metadata"].item()
    return matrices, metadata
