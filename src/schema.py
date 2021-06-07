from __future__ import annotations

from typing import List

from dataclasses import dataclass
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class SequenceLabelingExample:
    tokens: List[str]
    labels: List[str]

@dataclass_json
@dataclass
class TextClassificationExample:
    text: str
    label: str