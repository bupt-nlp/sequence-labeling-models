from typing import List, Iterator

from torch.nn import Parameter

from allen import (
    CCKSTriggerDatasetReader,
    CCKSTriggerSequenceLabelingDatasetReader,
    CCKSTriggerClassifier
)

from allennlp.data import Vocabulary, DataLoader, PyTorchDataLoader, \
    AllennlpDataset
from allennlp.training import GradientDescentTrainer
from allennlp.training.optimizers import HuggingfaceAdamWOptimizer, AdagradOptimizer


def build_data_loader(instances: AllennlpDataset) -> DataLoader:
    """构建数据加载起"""
    data_loader = PyTorchDataLoader(
        dataset=instances,batch_size=64,shuffle=True
    )
    return data_loader


def train():
    pre_trained_model_name = "bert-base-chinese"
    reader = CCKSTriggerSequenceLabelingDatasetReader(
        tokenizer=pre_trained_model_name,
        token_indexers=pre_trained_model_name
    )
    train_dataset: AllennlpDataset = reader.read('./data/train_base.json')
    vocab = Vocabulary.from_instances(train_dataset)
    train_dataset.index_with(vocab)

    # 构建DataLoader
    data_loader = build_data_loader(train_dataset)

    # # 开始构建model
    model = CCKSTriggerClassifier(
        vocab=vocab,
        model_name=pre_trained_model_name
    )
    parameters = [
        [n, p]
        for n, p in model.named_parameters() if p.requires_grad
    ]
    trainner = GradientDescentTrainer(
        model=model,
        optimizer=AdagradOptimizer(
            model_parameters=parameters,
        ),
        data_loader=data_loader,
        patience=10,
        num_epochs=20,
        serialization_dir='./output/trigger_BIEOS_classification',
        cuda_device=2
    )
    trainner.train()

    
if __name__ == "__main__":
    
    train()