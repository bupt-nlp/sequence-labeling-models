from __future__ import annotations

from typing import List, Union
from allennlp.data.dataset_readers.sequence_tagging import DEFAULT_WORD_TAG_DELIMITER


def read_line(line: str) -> Union[List[str], List[str]]: 
    """extract tokens&labels from one line<bmes>

    Args:
        line (str): one line: 北京 O

    Returns:
        Union[List[str], List[str]]: the result of tokens, labels
    """
    if len(line.split()) == 1:
        a = ''
    word, label = list(line.split())
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

        # 1. 一个文件中的token和labels
        sentence_tokens, sentence_labels = [], []
        for line in f:
            line = line.strip()
            if not line:
                sentences.append(sentence_tokens)
                labels.append(sentence_labels)
                sentence_tokens, sentence_labels = [], []
            else:
                line_tokens, line_labels = read_line(line)

                sentence_tokens.extend(line_tokens)
                sentence_labels.extend(line_labels)

    assert len(sentences) == len(labels)
    
    # 2. write tokens and labels to the file
    with open(output_file, 'w+', encoding='utf-8') as f:

        for index in range(len(sentences)):
            tokens, sentence_labels = sentences[index], labels[index]

            items = [
                '###'.join([tokens[i], sentence_labels[i]]) for i in range(len(tokens))]
                
            f.write('\t'.join(items) + '\n')
      