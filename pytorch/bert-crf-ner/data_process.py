import json
from tqdm import tqdm

from allennlp.data.tokenizers import PretrainedTransformerTokenizer
tokenizer = PretrainedTransformerTokenizer(model_name='bert-base-chinese')

from config import all_events


def generate_trigger_sequence_tokens(mode: str = 'train'):
    """对每个event生成对应的输入和输出"""

    mode_data = all_events[mode]

    with open(mode_data['train_file'], 'r+', encoding='utf-8') as f:

        trigger_sequence_file = open(
            mode_data['phrase_file'], 'w+',
            encoding='utf-8'
        )

        for line in tqdm(f.readlines()):
            line = line.replace('\n', '')
            data = json.loads(line)
            sentence_id = data['id']
            sentence = data['content']

            # 初始化所有事件的数据
            sentence_multi_events = {}
            for event in mode_data["labels"]:
                event_sentence = f'{sentence}[SEP]{event}'
                sentence_multi_events[event] = {
                    'sentence': event_sentence,
                    'tokenized_sentence': tokenizer.tokenize(event_sentence),
                    'trigger_words': [],

                }
                sentence_multi_events[event]['labels'] = ['O'] * len(sentence_multi_events[event]['tokenized_sentence'])

            for event in data.get('events', []):
                event_type = event['type']

                # 初始化新event事件数据
                mentions = event.get('mentions', [])
                trigger_mention = [mention for mention in mentions
                                   if mention['role'] == 'trigger']
                assert len(trigger_mention) == 1
                trigger_mention = trigger_mention[0]
                trigger_words_info = (
                    [trigger_mention['word'],
                     trigger_mention['span'][0],
                     trigger_mention['span'][1]]
                )
                sentence_multi_events[event_type]['trigger_words'].append(
                    trigger_words_info
                )

            # 对sentence 进行打标
            for event in mode_data['labels']:
                trigger_words = sorted(
                    sentence_multi_events[event]['trigger_words'],
                    key=lambda x: x[1]
                )
                tokenized_sentence = sentence_multi_events[event]['tokenized_sentence']
                for trigger_word in trigger_words:
                    # 开始对文本进行便利
                    word_len = len(trigger_word[0])
                    for sentence_index in range(len(tokenized_sentence) - word_len):
                        indexed_word = tokenized_sentence[sentence_index: sentence_index + word_len]
                        indexed_word = ''.join([word.text for word in indexed_word])
                        if indexed_word == trigger_word[0]:
                            # 找到了对应的位置
                            sentence_multi_events[event]['labels'][sentence_index] = 'B'
                            for i in range(sentence_index + 1, sentence_index + word_len):
                                sentence_multi_events[event]['labels'][i] = 'I'
                            break
            for event, value in sentence_multi_events.items():
                tokenized_sentence = value['tokenized_sentence']
                labels = value['labels']
                assert len(tokenized_sentence) == len(value['labels'])

                items = []
                for index in range(len(tokenized_sentence)):
                    items.append(
                        f'{tokenized_sentence[index].text}###{labels[index]}'
                    )
                items_str = '\t'.join(items)
                trigger_sequence_file.write(f'{items_str}\n')
        trigger_sequence_file.close()


def generate_role_sequence_tokens(mode: str = 'train'):
    """对不同事件来抽取不同的标签"""
    mode_data = all_events[mode]

    with open(mode_data['train_file'], 'r+', encoding='utf-8') as f:

        role_sequence_file = open(
            mode_data['role_file'], 'w+',
            encoding='utf-8'
        )

        for line in tqdm(f.readlines()):
            line = line.replace('\n', '')
            data = json.loads(line)
            sentence_id = data['id']
            sentence = data['content']



        role_sequence_file.close()


generate_trigger_sequence_tokens()