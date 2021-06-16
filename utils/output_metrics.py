from __future__ import annotations
import os
from tabulate import tabulate
from utils.config import ROOT_DIR

def output_metric(data, headers = None, file: str = './output/metrics.txt'):
    file = os.path.join(ROOT_DIR, file)
    with open('./output/metrics.txt', 'w+', encoding='utf-8') as f:
        f.write(tabulate(data, headers=headers, tablefmt='psql'))


if __name__ == '__main__':
    series = [[1, 2], [3, 4]]
    output_metric(series, ["a", "b"])