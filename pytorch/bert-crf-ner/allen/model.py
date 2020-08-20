from typing import Dict, Optional, Any
from pprint import pprint

import torch

from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules.seq2vec_encoders import BertPooler
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.training.metrics import CategoricalAccuracy, F1Measure, Metric
from allennlp.nn.util import get_text_field_mask


@Metric.register('multi-label-metric')
class MultiLabelMetric(Metric):

    def __call__(self, predictions: torch.Tensor, gold_labels: torch.Tensor):
        """计算多标签分类的准确率"""
        predictions[predictions>0.5] = 1.
        predictions[predictions<=0.5] = 0.
        batch_size, num_labels = predictions.shape
        pre: torch.Tensor = (predictions == gold_labels).sum()
        self.accuarcy = pre / (batch_size * num_labels)

    def get_metric(self, reset: bool) -> Dict[str, Any]:
        acc = self.accuarcy
        if reset:
            self.reset()
        return {"multi-acc": acc}

    def reset(self) -> None:
        self.accuarcy = 0.


@Model.register('ccks-trigger')
class CCKSTriggerClassifier(Model):
    """训练出模型的最终效果"""
    def __init__(self, vocab: Vocabulary, model_name: str):
        super().__init__(vocab=vocab)
        self.bert_pooler = BertPooler(pretrained_model=model_name)
        self.classifier_layer = torch.nn.Linear(
            in_features=self.bert_pooler.get_output_dim(),
            out_features=vocab.get_vocab_size('labels')
        )
        self.text_field_embedder = BasicTextFieldEmbedder(
            token_embedders={
                'tokens': PretrainedTransformerEmbedder(
                    model_name=model_name
                )
            }
        )
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.metrics = {
            "multi-label": MultiLabelMetric()
        }

    def forward(self, tokens: Dict[str, torch.Tensor],
                labels: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        """forward the input on sentence"""
        inputs: torch.Tensor = self.text_field_embedder(tokens)
        mask = get_text_field_mask(tokens)

        bert_output = self.bert_pooler(
            inputs,
            mask
        )
        logits = torch.nn.functional.softmax(self.classifier_layer(bert_output), dim=-1)
        output = {
            "logits": logits
        }
        if labels is not None:
            labels = labels.float()
            loss = self.loss_fn(logits, labels)
            output['loss'] = loss

            for name, metric in self.metrics.items():
                metric(logits, labels)
            # labels -> (batch_size, num_labels)
            # logits -> (batch_size, num_labels)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metric_result = {}
        for name, metric in self.metrics.items():
            all_metric_result.update(metric.get_metric(reset))
        return all_metric_result
