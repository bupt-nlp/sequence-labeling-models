#!/usr/bin/env python
# -*- encoding=utf-8 -*-
# author: ianma
# email: stmayue@gmail.com
# created time: 2020-07-20 10:32:34


import json
import random
import sys
from tqdm import tqdm

from collections import defaultdict

all_attributes = {
    "train": {
        "质押": ["sub-org", "sub-per", "obj-org", "obj-per", "collateral", "date",
               "money", "number", "proportion"],
        "股份股权转让": ["sub-org", "sub-per", "obj-org", "obj-per", "collateral",
                   "date", "money", "number", "proportion", "target-company"],
        "起诉": ["sub-org", "sub-per", "obj-org", "obj-per", "date"],
        "投资": ["sub", "obj", "money", "date"],
        "减持": ["sub", "obj", "title", "date", "share-per", "share-org"]
    },
    "test": {
        "担保": ["sub-org", "sub-per", "obj-org", "way", "amount", "date"],
        "中标": ["sub", "obj", "amount", "date"],
        "签署合同": ["sub-org", "sub-per", "obj-org", "obj-per", "amount", "date"]
    },
    "dev": {
        "收购": ["sub-org", "sub-per", "obj-org", "way", "date", "money",
               "number", "proportion"],
        "判决": ["institution", "sub-per", "sub-org", "obj-per", "obj-org",
               "date", "money"]
    }
}

files = {
    "train": "./data/train_base.json"
}


def classify_mentions(mentions):
    trigger_mention = None
    other_mentions = []
    for mention in mentions:
        if mention["role"] == "trigger":
            trigger_mention = mention
        else:
            other_mentions.append(mention)

    return trigger_mention, other_mentions


def main():
    data_type = 'train'

    sample_detail = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    attributes = all_attributes[data_type]
    with open(files[data_type], 'r+', encoding="utf-8") as f:
        lines = f.readlines()
    for line in tqdm(lines):
        line = line.strip()
        data = json.loads(line)
        sent = data["content"].replace('\r', '_').replace('\n', '_')
        sent_id = data["id"]

        for event in data["events"]:
            event_type = event["type"]
            all_roles = attributes[event_type][:]
            trigger_mention, other_mentions = classify_mentions(
                event["mentions"])
            trigger_word = trigger_mention["word"]
            trigger_start, trigger_end = trigger_mention["span"]
            # construct context
            context = sent[:trigger_start] + "<s>" + sent[
                                                     trigger_start:trigger_end] + "</s>" + sent[
                                                                                           trigger_end:]
            exist_role = set()
            for mention in other_mentions:
                current_role = mention["role"]
                current_word = mention["word"]
                current_start, current_end = mention["span"]
                exist_role.add(current_role)
                if current_role not in all_roles:
                    print(event_type, current_role)
                    sys.exit(-1)

                if current_start > trigger_start and current_start >= trigger_end:
                    current_start += 7
                    current_end += 7
                elif current_start > trigger_start:
                    sys.stderr.write(
                        "{}\t{}\t{}\n".format(context, current_word,
                                              current_start))
                    continue

                sample_detail[context][event_type][current_role].append(
                    [current_word, current_start, current_end])

            for role in all_roles:
                if role in exist_role:
                    continue
                sample_detail[context][event_type][role] = []

    for context in sample_detail:
        for event_type in sample_detail[context]:
            for role in sample_detail[context][event_type]:
                ans = sample_detail[context][event_type][role]
                ans = ";".join(map(lambda x: "|".join(map(str, x)), ans))
                print("{}\t{}\t{}\t{}".format(context, event_type, role, ans))

        for event_type in attributes:
            if event_type in sample_detail[context]:
                continue
            if random.random() < 0.5:
                continue
            for role in attributes[event_type]:
                if random.random() > 0.5:
                    continue
                print("{}\t{}\t{}\t{}".format(context, event_type, role, ""))


def transfer_file_to_allennlp_sequence_tagging_data_format():
    """将处理好的数据转换成"""
    with open('./data/train_base_role.txt', 'r+', encoding='utf-8') as f:
        role_sequence_tagging_file = open('./data/train_base_role_tagging.txt', 'w+', encoding='utf-8')
        from allennlp.data.tokenizers import PretrainedTransformerTokenizer
        tokenizer = PretrainedTransformerTokenizer(model_name='bert-base-chinese')

        no_recongnized_types = []

        for line in tqdm(f.readlines()):
            line = line.strip()
            items = line.split('\t')

            sentence = f'{items[0].replace(" ","")}[SEP]{items[1]};{items[2]}'
            tokens = tokenizer.tokenize(items[0])
            labels = ['O'] * len(tokens)

            if len(items) == 4:
                assert len(items) == 4
                role_attribute_list = items[3].split(';')

                for role_attribute in role_attribute_list:
                    attributes_str = role_attribute.split('|')
                    role_words = attributes_str[0]
                    role_words_tokenized = tokenizer.tokenize(role_words)[1:-1]
                    role_words_tokenized_str = ''.join([item.text for item in role_words_tokenized])
                    role_words_len = len(role_words_tokenized)
                    # 查找目标words
                    find_the_words = False
                    words_start, words_end = -1, -1
                    for index in range(len(tokens) - role_words_len):
                        if find_the_words:
                            break
                        find_tokens = tokens[index: index+role_words_len]
                        finded_words = ''.join([token.text for token in find_tokens])
                        if finded_words.replace(' ','') == role_words_tokenized_str.replace(' ', ''):
                            find_the_words = True
                            words_start, words_end = index, index + role_words_len

                    # 开始个对应的位置打上标签
                    if not find_the_words or words_end == -1 or words_start == -1:
                        # bert 不识别的话，那也没办法
                        # print('not recongnized ...')
                        # print(items)
                        no_recongnized_types.append(items[2])
                        continue
                        # print(items[2])
                        # print('未在文本中查找到对应的属性词')

                    labels[words_start] = "B"
                    for index in range(words_start + 1, words_end):
                        labels[index] = "I"

            all_tags = []
            for index in range(len(tokens)):
                all_tags.append(f'{tokens[index].text}###{labels[index]}')

            all_tags_str = '\t'.join(all_tags)
            role_sequence_tagging_file.write(all_tags_str + '\n')

        print(f'有{len(no_recongnized_types)}条数据未能识别')
        print(set(no_recongnized_types))
        print('以下类型无法识别：')
        print(set(no_recongnized_types))



if __name__ == "__main__":
    # main()
    transfer_file_to_allennlp_sequence_tagging_data_format()