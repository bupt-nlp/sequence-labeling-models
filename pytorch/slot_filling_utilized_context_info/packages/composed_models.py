from typing import Dict

import torch
from allennlp.models import Model
from allennlp.data import Vocabulary


@Model.register('composed_model')
class ComposedModel(Model):
    def __init__(self, vocab: Vocabulary, ):
        super().__init__(vocab)

    def forward(self, tokens, labels) -> Dict[str, torch.Tensor]:
        return self.one(tokens, labels)

