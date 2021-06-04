from __future__ import annotations

from typing import List, Union


def read_line(line: str) -> Union[List[str], List[str]]: 
    word, label = list(line.split(' '))
    tokens = list(word)
    if label == 'O':
        return tokens, ['O']* len(tokens)

    labels = []

    labels.append(f'B-{label}')
    if len(tokens) == 1:
        return tokens, labels
    if len(tokens) == 2:
        labels.append(f'E-{label}')
        return tokens, labels

    for i in range(1, len(tokens)-1):
        labels.append(f'I-{label}')
    labels.append(f'E-{label}')
    assert len(labels) == len(tokens)

    return tokens, labels 
    

def convert_bmes_to_sequence_tagging(source_file: str, output_file: str):
    """convert_bmes_to_sequence_tagging convert bbmes format data to sequence-tagging data format

    Args:
        source_file (str): the path of bmes format file
        output_file (str): the output file
    """
    # 1. read all lines and split it to sentences
    sentences: List[str] = []
    labels: List[str] = []
    with open(source_file, 'r+', encoding='utf-8') as f:
        temp_tokens, temp_labels = [], []
        for line in f:
            if not line:
                sentences.append(temp_tokens)
                labels.append(temp_labels)
                temp_tokens, temp_labels = [], []
            else:
                
                
                
            
        