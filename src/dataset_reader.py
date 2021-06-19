"""
dataset reader
"""

from allennlp.data.dataset_readers import DatasetReader
from transformers import BertTokenizerFast


class JointDatasetReader(DatasetReader):
    def __init__(self, tokenizer: BertTokenizerFast):
        self.tokenizer = tokenizer

    def _read_intents(self):
        pass

    def read_slots(self):
        pass

    def _read(self):
        pass
    