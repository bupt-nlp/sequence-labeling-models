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
    TextFieldTensors,
)
from allennlp.data.data_loaders import SimpleDataLoader
from allennlp.models import Model
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.training.optimizers import AdamOptimizer

from utils.dataset_processor import read_line, convert_bmes_to_sequence_tagging


def test_read_line():
    line = '我爱 O'
    tokens, labels = read_line(line)
    assert tokens == ['我', '爱']
    assert labels == ['O', 'O']

    line = '吴京 PER'
    tokens, labels = read_line(line)
    assert tokens == ['吴', '京']
    assert labels == ['B-PER', 'E-PER']

    line = '吴京京 PER'
    tokens, labels = read_line(line)
    assert tokens == ['吴', '京', '京']
    assert labels == ['B-PER', 'M-PER', 'E-PER']
    
def test_bmes_converter():
    base_dir = './data/weibo'
    for file in os.listdir(base_dir):
        if not file.endswith('.txt'):
            continue

        input_file = os.path.join(base_dir, file)
        output_file = os.path.join(base_dir, file.replace('txt', 'corpus'))
        convert_bmes_to_sequence_tagging(input_file, output_file)

def test_sequence_tagging_reader():
    model_name = 'bert-base-chinese'

    bert_token_indexers = PretrainedTransformerIndexer(model_name=model_name)
    reader = SequenceTaggingDatasetReader(token_indexers={"tokens": bert_token_indexers})

    train_file = './data/weibo/train.corpus'
    dev_file = './data/weibo/dev.corpus'
    test_file = './data/weibo/dev.corpus'
    train_instances = list(reader.read(train_file))
    dev_instances = list(reader.read(dev_file))
    test_instances = list(reader.read(test_file))

    vocab: Vocabulary = Vocabulary.from_instances(train_instances)
    assert vocab.get_namespaces() is not None

    bert_text_field_embedder = PretrainedTransformerEmbedder(model_name=model_name)
    tagger = SimpleTagger(
        vocab=vocab,
        text_field_embedder=BasicTextFieldEmbedder(
            token_embedders={
                'tokens': bert_text_field_embedder
            }
        ),
        encoder=PassThroughEncoder(bert_text_field_embedder.get_output_dim()),
        calculate_span_f1=True,
        label_encoding="BMES",
        # verbose_metrics=True
    )

    train_data_loader, dev_data_loader = build_data_loaders(train_instances, dev_instances)
    train_data_loader.index_with(vocab)
    dev_data_loader.index_with(vocab)

    trainer = build_trainer(model=tagger, serialization_dir='./output', train_loader=train_data_loader, dev_loader=dev_data_loader)
    print("Starting training")
    trainer.train()
    print("Finished training")


def build_data_loaders(
    train_data: List[Instance],
    dev_data: List[Instance],
) -> Tuple[DataLoader, DataLoader]:
    train_loader = SimpleDataLoader(train_data, 8, shuffle=True)
    dev_loader = SimpleDataLoader(dev_data, 8, shuffle=False)
    return train_loader, dev_loader


def build_trainer(
    model: Model,
    serialization_dir: str,
    train_loader: DataLoader,
    dev_loader: DataLoader,
) -> Trainer:
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    optimizer = AdamOptimizer(parameters)  # type: ignore
    trainer = GradientDescentTrainer(
        model=model,
        serialization_dir=serialization_dir,
        data_loader=train_loader,
        validation_data_loader=dev_loader,
        num_epochs=5,
        optimizer=optimizer,
    )
    return trainer

if __name__ == "__main__":
    test_sequence_tagging_reader()