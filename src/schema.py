from __future__ import annotations

from typing import List

from dataclasses import dataclass
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class SequenceLabelingExample:
    def __init__(self, tokens: List[str], labels: List[str]):
        assert len(tokens) == len(labels)
        self.tokens = tokens
        self.labels = labels

    tokens: List[str]
    labels: List[str]
