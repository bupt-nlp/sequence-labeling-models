import pytest
from transformers import BertModel, BertTokenizerFast, BertConfig
from src.models.joint_bert import JointBERT, JointBertConfig

@pytest.fixture
def model_name():
    return 'bert-base-uncased'

@pytest.fixture
def config(model_name):
    return BertConfig.from_pretrained(model_name)

@pytest.fixture
def tokenizer(model_name):
    return BertTokenizerFast.from_pretrained(model_name)


def test_bert_model(config: BertConfig, tokenizer: BertTokenizerFast):
    model = BertModel(config)
    sentence = "I lovvvvvve china"
    output = tokenizer(sentence, return_offsets_mapping=True, is_split_into_words=True)

def test_joint_bert(config: BertConfig, tokenizer: BertTokenizerFast):
    joint_model_config: JointBertConfig = JointBertConfig().parse_args(known_only=True)
    joint_model_config.bert_config = config

    joint_model = JointBERT(joint_model_config)

    sentence = 'i love china'
    input_data = tokenizer(sentence, return_tensors="pt")
    output = joint_model(**input_data)
    assert output is not None
