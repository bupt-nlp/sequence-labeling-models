from __future__ import annotations

import os
from typing import List, Tuple

from allennlp.data import Vocabulary, Instance
from allennlp.data.data_loaders import SimpleDataLoader, DataLoader
from allennlp.data.token_indexers.pretrained_transformer_indexer import PretrainedTransformerIndexer
from allennlp.data.dataset_readers.sequence_tagging import SequenceTaggingDatasetReader
from allennlp.models.model import Model

from allennlp.models.simple_tagger import SimpleTagger
from allennlp.modules.text_field_embedders.basic_text_field_embedder import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.pretrained_transformer_embedder import PretrainedTransformerEmbedder
from allennlp.modules.seq2seq_encoders import PassThroughEncoder

from allennlp.training.trainer import Trainer
from allennlp.training import GradientDescentTrainer

import torch
from allennlp.data import (
    DataLoader,
    DatasetReader,
    Instance,
    Vocabulary,
)
from allennlp.data.data_loaders import SimpleDataLoader
from allennlp.models import Model
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.training.optimizers import AdamOptimizer

from loguru import logger
from transformers.utils.dummy_pt_objects import ConvBertForTokenClassification
from tap import Tap

class Config(Tap):
    device: int = 1
    
    model_name: str = 'hfl/chinese-bert-wwm-ext'
    train_file: str = './data/weibo/train.corpus'
    dev_file: str = './data/weibo/dev.corpus'
    test_file: str = './data/weibo/dev.corpus'
    
    batch_size: int = 32
    epoch: int = 20
    lr: float = 0.005
    

class TaggerTrainer:
    def __init__(self) -> None:
        self.config: Config = Config().parse_args(known_only=True)
        
        bert_token_indexers = PretrainedTransformerIndexer(model_name=self.config.model_name)
        reader = SequenceTaggingDatasetReader(token_indexers={"tokens": bert_token_indexers})

        train_instances = list(reader.read(self.config.train_file))
        dev_instances = list(reader.read(self.config.dev_file))
        test_instances = list(reader.read(self.config.test_file))

        self.vocab: Vocabulary = Vocabulary.from_instances(train_instances)

        # 2. init the data loader
        self.train_data_loader = SimpleDataLoader(train_instances, self.config.batch_size, shuffle=True)
        self.dev_data_loader = SimpleDataLoader(dev_instances, self.config.batch_size, shuffle=False)
        self.train_data_loader.index_with(self.vocab)
        self.dev_data_loader.index_with(self.vocab)
        
        # 3. init the model
        self.model = self.init_model()
        self.trainer = self.init_trainer()
        
    
    def init_model(self) -> Model:
        """build the model

        Args:
            vocab (Vocabulary): the vocabulary of corpus

        Returns:
            Model: the final models
        """
        bert_text_field_embedder = PretrainedTransformerEmbedder(model_name=self.config.model_name)
        bert_text_field_embedder
        tagger = SimpleTagger(
            vocab=self.vocab,
            text_field_embedder=BasicTextFieldEmbedder(
                token_embedders={
                    'tokens': bert_text_field_embedder
                }
            ),
            encoder=PassThroughEncoder(bert_text_field_embedder.get_output_dim()),
            verbose_metrics=True,
            calculate_span_f1=True,
            label_encoding="BMES",
        )
        
        tagger.to(device=self.config.device)
        return tagger
    
    def init_trainer(self) -> Trainer:
        parameters = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        optimizer = AdamOptimizer(parameters, lr=self.config.lr)  # type: ignore
        trainer = GradientDescentTrainer(
            model=self.model,
            serialization_dir='./output',
            data_loader=self.train_data_loader,
            validation_data_loader=self.dev_data_loader,
            num_epochs=self.config.epoch,
            optimizer=optimizer,
            cuda_device=self.config.device,
        )
        return trainer
    
    def train(self):
        self.trainer.train()
    
    
if __name__ == '__main__':
    trainer = TaggerTrainer()
    trainer.train()
