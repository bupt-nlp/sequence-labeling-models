import os
import pickle
from tqdm import trange


def load_ds(file: str):
    with open(file, 'rb') as stream:
        ds, dicts = pickle.load(stream)
    print('Done  loading: ', file)
    print('      samples: {:4d}'.format(len(ds['query'])))
    print('   vocab_size: {:4d}'.format(len(dicts['token_ids'])))
    print('   slot count: {:4d}'.format(len(dicts['slot_ids'])))
    print(' intent count: {:4d}'.format(len(dicts['intent_ids'])))
    return ds, dicts


def generate_sequence_tagging_data(mode: str = "train"):
    """生成指定的序列标注的模型"""
    dataset, dicts = load_ds(f'./data/atis.{mode}.pkl')
    queries = dataset['query']
    slot_labels = dataset['slot_labels']

    token_id2idx = {idx: word for word,idx in dicts['token_ids'].items()}
    slot_id2idx = {idx: slot for slot,idx in dicts['slot_ids'].items()}

    tagging_data_dir = './label_data'
    if not os.path.exists(tagging_data_dir):
        os.mkdir(tagging_data_dir)

    tagging_file = open(f'{tagging_data_dir}/label_{mode}.txt', 'w+', encoding='utf-8')

    for index in trange(len(queries)):
        tokens = [token_id2idx[item] for item in queries[index]]
        slots = [slot_id2idx[item] for item in slot_labels[index]]

        assert len(tokens) == len(slots)

        items = [f'{tokens[i]}###{slots[i]}' for i in range(len(tokens))]
        tagging_file.write('\t'.join(items) + '\n')

    tagging_file.close()


generate_sequence_tagging_data()