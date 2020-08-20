
import torch
from transformers import BertForTokenClassification, BertTokenizer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer


def build_bert_www_model(model_name: str = "hfl/chinese-bert-wwm-ext") -> torch.nn.Module:

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForTokenClassification.from_pretrained(model_name)

    inputs = tokenizer(["Hello, my dog is cute","Hello, my dog is cute"], return_tensors="pt")

    labels = torch.tensor([1] * inputs["input_ids"].size(1)).unsqueeze(
        0)  # Batch size 1
    outputs = model(**inputs, labels=labels)
    loss, scores = outputs[:2]

    return model

def two_sentence_tokenize(sentence_1, sentence_2):
    tokenizer = PretrainedTransformerTokenizer(model_name='bert-base-chinese')
    from pprint import pprint
    sen_1 = tokenizer.tokenize(sentence_1)
    sen_2 = tokenizer.tokenize(sentence_2)

    pprint(sen_1)
    pprint(sen_2)


two_sentence_tokenize('我爱你中国', '好好')