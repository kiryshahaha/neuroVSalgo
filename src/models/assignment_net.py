import torch
import torch.nn as nn
from src.models.graph_edge_mlp import GraphContextEdgeMLP
from src.models.sinkhorn import SinkhornLayer
from src.models.fast_decoder import FastGpuGreedyDecoder


class NeuralAssignmentModel(nn.Module):
    """
    Модернизированная сквозная нейросетевая модель с:
    1. GraphContextEdgeMLP (Message passing + InstanceNorm).
    2. SinkhornLayer (Log-domain нормализация).
    3. FastGpuGreedyDecoder (Параллельный GPU декодер без медленного numpy argsort).
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        sinkhorn_iters: int = 25,
        tau: float = 0.08,
    ):
        super().__init__()
        self.encoder = GraphContextEdgeMLP(hidden_dim=hidden_dim)
        self.sinkhorn = SinkhornLayer(num_iters=sinkhorn_iters, tau=tau)
        self.decoder = FastGpuGreedyDecoder()

    def forward(self, cost_matrix: torch.Tensor) -> torch.Tensor:
        """
        Дифференцируемый прямой проход.
        Меньшая стоимость должна получать больший приоритет, поэтому передаем -cost_matrix.
        """
        logits = self.encoder(-cost_matrix)
        prob_matrix = self.sinkhorn(logits)
        return prob_matrix

    @torch.no_grad()
    def predict(self, cost_matrix: torch.Tensor):
        """
        Высокоскоростной инференс с GPU-декодированием.
        """
        prob_matrix = self.forward(cost_matrix)
        assignments = self.decoder.decode_tensor(prob_matrix)
        return assignments, prob_matrix
