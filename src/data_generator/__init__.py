from typing import Optional, Literal
from src.data_generator.base import BaseDataGenerator
from src.data_generator.synthetic import UniformGenerator, LognormalGenerator
from src.data_generator.spatial import Spatial2DGenerator
from src.data_generator.io import save_dataset, load_dataset

DistributionType = Literal["uniform", "lognormal", "spatial_2d", "spatial_manhattan"]


def get_generator(
    dist_type: DistributionType,
    seed: Optional[int] = None,
    **kwargs,
) -> BaseDataGenerator:
    """
    Фабричный метод для инстанцирования нужного типа генератора.

    :param dist_type: Тип распределения ("uniform", "lognormal", "spatial_2d", "spatial_manhattan").
    :param seed: Зерно псевдослучайного генератора.
    :param kwargs: Дополнительные параметры для конкретных генераторов.
    :return: Экземпляр соответствующего генератора.
    """
    dist_type = dist_type.lower()
    if dist_type == "uniform":
        low = kwargs.get("low", 0.0)
        high = kwargs.get("high", 1.0)
        return UniformGenerator(low=low, high=high, seed=seed)

    elif dist_type == "lognormal":
        mean = kwargs.get("mean", 0.0)
        sigma = kwargs.get("sigma", 0.75)
        normalize_max = kwargs.get("normalize_max", False)
        return LognormalGenerator(mean=mean, sigma=sigma, normalize_max=normalize_max, seed=seed)

    elif dist_type == "spatial_2d":
        bounds = kwargs.get("bounds", (0.0, 1.0, 0.0, 1.0))
        return Spatial2DGenerator(bounds=bounds, metric="euclidean", seed=seed)

    elif dist_type == "spatial_manhattan":
        bounds = kwargs.get("bounds", (0.0, 1.0, 0.0, 1.0))
        return Spatial2DGenerator(bounds=bounds, metric="manhattan", seed=seed)

    else:
        raise ValueError(
            f"Неизвестный тип распределения: {dist_type}. "
            f"Допустимые: 'uniform', 'lognormal', 'spatial_2d', 'spatial_manhattan'."
        )


__all__ = [
    "BaseDataGenerator",
    "UniformGenerator",
    "LognormalGenerator",
    "Spatial2DGenerator",
    "get_generator",
    "save_dataset",
    "load_dataset",
]
