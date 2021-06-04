from __future__ import annotations

import os
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
    assert labels == ['B-PER', 'I-PER', 'E-PER']
    
def test_bmes_converter():
    base_dir = './data/weibo'
    for file in os.listdir(base_dir):
        if not file.endswith('.txt'):
            continue

        input_file = os.path.join(base_dir, file)
        output_file = os.path.join(base_dir, file.replace('txt', 'corpus'))
        convert_bmes_to_sequence_tagging(input_file, output_file)
