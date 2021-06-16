from __future__ import annotations

from tabulate import tabulate

def test_bert_crf():
    table = [["Sun",696000,1989100000],["Earth",6371,5973.6],
          ["Moon",1737,73.5],["Mars",3390,641.85]]
    with open('./output/metrics.txt', 'w+', encoding='utf-8') as f:
        f.write(tabulate(table))