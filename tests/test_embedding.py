from __future__ import annotations

import torch
from src.models.global_pointer import SinusoidalPositionEmbedding


def test_embedding():
    input_embedding = torch.randn((16, 32, 64))
    embedding = SinusoidalPositionEmbedding()

    