"""
Author: <wj-Mcat> 吴京京
Time: 2020-08-15
Email: 1435130236@qq.com

Standard dataset reader for sequance-labeling task. 
If you can follow the sequance-labeling training data structure, 
you can directly use this dataset reader.
"""
from __future__ import annotations

import json
from typing import Iterable, Optional, Dict, List
from pprint import pprint

from allennlp.data import DatasetReader, Instance, Tokenizer, TokenIndexer, \
    Token
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.token_indexers import (
    SingleIdTokenIndexer,
    PretrainedTransformerIndexer
)

from allennlp.data.fields import (
    SpanField, TextField,
    SequenceLabelField,
    MultiLabelField
)

from config import all_events

from allennlp_utils.data.tokenizer import JiebaTokenizer


import torch
torch.nn.CrossEntropyLoss()

class Mention:
    def __init__(self, word: str, span: List[int], role: str):
        self.word = word
        self.span = span
        self.role = role


@DatasetReader.register('ccks-phrase')
class CCKSPhraseDatasetReader(DatasetReader):
    def __init__(self, tokenizer: str,
                 token_indexers: str):
        super().__init__()
        self.tokenizer = PretrainedTransformerTokenizer(model_name=tokenizer)
        self.token_indexers = {
            'tokens': PretrainedTransformerIndexer(token_indexers)
        }

    def _read(self, file_path: str) -> Iterable[Instance]:
        file_path = all_events[file_path]['phrase_file']
        with open(file_path, 'r+', encoding='utf-8') as f:
            for line in f:
                data = line.replace('\n', '').split('\t')
                assert len(data) == 2

                tokens = self.tokenizer.tokenize(data[0])
                assert len(tokens) + 2 == len(data[1])

                labels = f'O{data[1]}O'.split()

                yield self.text_to_instance(tokens, labels)

    def text_to_instance(self, tokens: List[Token],
                         labels: Optional[List[str]] = None
                         ) -> Instance:
        fields = {
            "tokens": TextField(tokens, token_indexers=self.token_indexers)
        }
        if labels:
            fields['labels'] = SequenceLabelField(
                labels=labels,
                sequence_field=fields['tokens']
            )
        return Instance(fields)