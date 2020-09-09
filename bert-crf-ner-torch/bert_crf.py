"""
Author : http://www.github.com/wj-Mcat
Date : 2020.09.09
"""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import json
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from transformers import (
    CONFIG_NAME,
    WEIGHTS_NAME,
    BertConfig,
    BertForTokenClassification,
    BertTokenizer
)

from seqeval.metrics import classification_report, f1_score
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from .crf import ConditionalRandomField

import torch.utils.data as Data

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class Ner(BertForTokenClasswification):
    """
    custom the transformer BertForTokenClassification Model

    transformer Bert Model can simplify the bert implemention code
    """
    def __init__(self,
                 config, num_labels: int,
                 num_features: int = 10,
                 feature_embedding_dim: int = 40,
                 use_crf: bool = True):
        """
        Args:
            config: this is the configuration for BertModel, which is required to init the model
            num_labels: this is this is defined for the 
        """
        super(Ner, self).__init__(config)

        # 在最后一层，添加一个线性转化层即可
        self.feature_embedding_lookup = torch.nn.Embedding(num_features, feature_embedding_dim)

        if use_crf:

        self.merge_classifier = nn.Linear(config.hidden_size + feature_embedding_dim, num_labels)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids: torch.Tensor, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None,
                attention_mask_label=None, fea_ids=None):

        # 使用transformers 的模型可以非常简单的调用bert
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        batch_size, max_len, feat_dim = sequence_output.shape

        feature_embedding = self.label_embedding_lookup(fea_ids)

        contat_output = torch.cat((sequence_output, feature_embedding), 2)
        # logits = self.classifier(sequence_output)
        # 使用线性层对数据进行融合和转化
        logits = self.merge_classifier(contat_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            attention_mask_label = None
            if attention_mask_label is not None:
                # attention_mask_label -> (batch_size, max_sequence_length, hidden_size)
                # 直接将mask给拉平，在一定程度上也是可以加速计算的
                active_loss_mask = attention_mask_label.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss_mask]
                active_labels = labels.view(-1)[active_loss_mask]

                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class InputExample(object):
    """A single training/test example for simple sequence classification.
    ： 数据的流程化 & 规范化

    处理和bert是一致，减少torch和tf开发者的过渡期
    """

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """
    A single set of features of data.

    这里实际上也是应该添加更多的类型标注，不然缺少一定的代码层级的语义信息
    """

    def __init__(self, input_ids, input_mask, segment_ids, label_id, valid_ids=None, label_mask=None, fea_ids=None,
                 raw_data=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask
        self.fea_ids = fea_ids
        self.raw_data = raw_data


def readfile(filename):
    '''
    read file
        format:
            我 O PER
            爱 O O
            你 O PER
    return format :
    [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
    '''
    print(filename)
    f = open(filename, 'r', encoding="utf-8")
    data = []
    sentence = []
    fea = []
    label = []
    for line in f:
        try:
            # print(line)
            if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
                if len(sentence) > 0:
                    data.append((sentence, fea, label))
                    sentence = []
                    label = []
                    fea = []
                continue
            # example data ： Swedish O B-LOC
            splits = line.split(' ')
            sentence.append(splits[0])

            if len(splits) == 2:
                # fea 表示
                fea.append(splits[-1][:-1])
            else:
                fea.append(splits[-2])
            label.append(splits[-1][:-1])
        except Exception as e:
            pass

    if len(sentence) > 0:
        data.append((sentence, fea, label))
        sentence = []
        label = []
        fea = []
    return data


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return readfile(input_file)


class NerProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(data_dir), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(data_dir), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(data_dir), "test")

    def get_labels(self):
        """获取序列标注任务当中所有的labels标签
        """
        return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "[CLS]", "[SEP]"]

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (sentence, fea, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            # label 标签的信息融入进去还是非常有必要的
            text_a = ' '.join(sentence)
            text_b = fea
            label = label
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list, 1)}

    features = []
    raw_data = []
    tot = 0
    for (ex_index, example) in enumerate(examples):
        textlist = example.text_a.split(' ')
        fealist = example.text_b
        labellist = example.label
        tokens = []
        labels = []
        valid = []
        label_mask = []
        feas = []
        raw_data.append(textlist)
        for i, word in enumerate(textlist):

            # 再针对于单个词语进行深度的分词
            token = tokenizer.tokenize(word)
            tokens.extend(token)

            label_1 = labellist[i]
            # fealist 是外部融合的知识
            label_0 = fealist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                    feas.append(label_0)
                    # valid 用来选择有效的文本数据ß
                    valid.append(1)
                    label_mask.append(1)
                else:
                    valid.append(0)

        # 需要根据max-length对文本进行截断
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            feas = feas[0:(max_seq_length - 2)]
            valid = valid[0:(max_seq_length - 2)]
            label_mask = label_mask[0:(max_seq_length - 2)]

        ntokens = []
        segment_ids = []
        label_ids = []
        fea_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        valid.insert(0, 1)
        label_mask.insert(0, 1)
        label_ids.append(label_map["[CLS]"])
        fea_ids.append(label_map["[CLS]"])

        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            if len(labels) > i:
                # print(labels[i])
                try:
                    label_ids.append(label_map[labels[i]])
                    fea_ids.append(label_map[feas[i]])
                except Exception as e:
                    print("ERROR", e)
                    # print(tokens)
                    # print(labels)
                    exit(0)
        ntokens.append("[SEP]")
        segment_ids.append(0)
        valid.append(1)
        label_mask.append(1)
        label_ids.append(label_map["[SEP]"])
        fea_ids.append(label_map["[SEP]"])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        label_mask = [1] * len(label_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            fea_ids.append(0)
            valid.append(1)
            label_mask.append(0)
        while len(label_ids) < max_seq_length:
            label_ids.append(0)
            label_mask.append(0)
        while len(fea_ids) < max_seq_length:
            fea_ids.append(0)
        assert len(input_ids) == max_seq_length
        assert len(fea_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(valid) == max_seq_length
        assert len(label_mask) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            # logger.info("label: %s (id = %d)" % (example.label, label_ids))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_ids,
                          valid_ids=valid,
                          label_mask=label_mask,
                          fea_ids=fea_ids,
                          raw_data=[tot]))
        tot += 1
    return features, raw_data


def main():
    import sys, os
    args = sys.argv
    parser = argparse.ArgumentParser()

    os.environ['CUDA_VISIBLE_DEVICES'] = "2"
    ## Required parameters
    parser.add_argument("--train_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The training dataset file.")
    parser.add_argument("--dev_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The development dataset file.")
    parser.add_argument("--test_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The test dataset file.")
    parser.add_argument("--pred_file",
                        default=None,
                        type=str,
                        required=False,
                        help="The output file where the model predictions will be written.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--early_stop",
                        default=5,
                        type=int,
                        help="Early etop")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_dir",
                        default=None,
                        type=str,
                        help="The directory of the model that need to be loaded.")
    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()

    # 要想对模型进行训练，还是需要使用shell脚本要来的更方便一些

    # 这部分就没有多大的必要了
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    processors = {"ner": NerProcessor}

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()
    # n + 1 中标签，这样能够识别出非目标标签
    num_labels = len(label_list) + 1

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.train_file)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

        # 分布式计算所需要的知识点
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                   'distributed_{}'.format(args.local_rank))
    model = Ner.from_pretrained(args.bert_model,
                                cache_dir=cache_dir,
                                num_labels=num_labels)
    if args.load_dir:
        # 如果文件夹下有相关的文件，就直接加载对应的权重文件
        output_config_file = os.path.join(args.load_dir, CONFIG_NAME)
        output_model_file = os.path.join(args.load_dir, WEIGHTS_NAME)
        config = BertConfig(output_config_file)
        model = Ner(config, num_labels=num_labels)
        model.load_state_dict(torch.load(output_model_file))

    # 这个配置就暂时不需要使用了，因为可能涉及到系统的不同配置
    if args.fp16:
        model.half()

    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        # 使用第三方库来进行分布式计算
        model = DDP(model)

    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    label_map = {i: label for i, label in enumerate(label_list, 1)}
    if args.do_train:
        # 将输入的examples转化成features
        train_features, raw_ = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_valid_ids = torch.tensor([f.valid_ids for f in train_features], dtype=torch.long)
        all_lmask_ids = torch.tensor([f.label_mask for f in train_features], dtype=torch.long)
        all_fea_ids = torch.tensor([f.fea_ids for f in train_features], dtype=torch.long)

        # 使用TensorDataset对数据进行打包
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_valid_ids,
                                   all_lmask_ids, all_fea_ids)

        if args.local_rank == -1:
            # sampler 随机采样
            train_sampler = RandomSampler(train_data)
        else:
            # 基于分布式策略的采样方法
            train_sampler = DistributedSampler(train_data)

        # 这一套东西都是使用pytorch，也可以与transformers的东西进行无缝融合
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        train_log = open(os.path.join(args.output_dir, "train.log"), "w")
        model.train()
        best_f1_score = -1
        best_round = 0

        # 这里可以设置range中的部分描述
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):

                # 批量加载到cuda上面去，可以有效减少cuda的现存使用量
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask, fea_ids = batch
                # 将数据塞入到模型当中
                loss = model(input_ids, segment_ids, input_mask, label_ids, valid_ids, l_mask, fea_ids)

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                print("epoch=%d, step=%d, loss=%.5f" % (_, step, loss), file=train_log, flush=True)
                tr_loss += loss.item()

                # 计算训练样本数据的数量
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_optimization_steps,
                                                                          args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            # 一个epoch过了之后就需要开始进行准确度验证，这个过程尽量是手动来操作，不过大部分的还是直接copy代码要来的干脆直接，也不容易出错
            # 即使是出错了，需要调整的地方也是很少
            # 我推荐还是自己掌握几套关于不同方向上的代码，这样能够在处理不同任务上能够很快的上手解决。
            # 这个过程就好像类似于现在code-generator

            print("Start Evaluating epoch %d ..." % _)
            eval_examples = processor.get_dev_examples(args.dev_file)

            eval_features, raw_ = convert_examples_to_features(eval_examples, label_list, args.max_seq_length,
                                                               tokenizer)
            logger.info("***** Running dev evaluation *****")
            logger.info("  Num examples = %d", len(eval_examples))
            logger.info("  Batch size = %d", args.eval_batch_size)

            all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
            all_valid_ids = torch.tensor([f.valid_ids for f in eval_features], dtype=torch.long)
            all_lmask_ids = torch.tensor([f.label_mask for f in eval_features], dtype=torch.long)
            all_fea_ids = torch.tensor([f.fea_ids for f in eval_features], dtype=torch.long)
            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_valid_ids,
                                      all_lmask_ids, all_fea_ids)

            # Run prediction for full data
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
            model.eval()

            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            y_true = []
            y_pred = []
            label_map = {i: label for i, label in enumerate(label_list, 1)}
            for input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask, fea_ids in tqdm(eval_dataloader,
                                                                                                  desc="DEV_Evaluating"):
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                valid_ids = valid_ids.to(device)
                label_ids = label_ids.to(device)
                l_mask = l_mask.to(device)
                fea_ids = fea_ids.to(device)

                with torch.no_grad():
                    logits = model(input_ids, segment_ids, input_mask, valid_ids=valid_ids, attention_mask_label=l_mask,
                                   fea_ids=fea_ids)

                logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)

                # 将预测数据的结果迁移到cpu上面进行计算
                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()

                # mask 的数据还是需要进行好好计算的
                input_mask = input_mask.to('cpu').numpy()

                # 开始对label数据进行验证
                for i, label in enumerate(label_ids):
                    temp_1 = []
                    temp_2 = []
                    # 第一个label为 [CLS]，所以是不需要进行计算其标签的。
                    for j, m in enumerate(label):
                        if j == 0:
                            continue

                        # 表示是最后一个标签
                        elif label_ids[i][j] == 11:
                            y_true.append(temp_1)
                            y_pred.append(temp_2)
                            break
                        else:
                            tmp_label = label_map.get(label_ids[i][j], "O")
                            # if "MISC" in tmp_label:
                            #    tmp_label = "O"
                            temp_1.append(tmp_label)

                            tmp_label = label_map.get(logits[i][j], "O")
                            # if "MISC" in tmp_label:
                            #    tmp_label = "O"
                            temp_2.append(tmp_label)

            # 将预测的结果报告输入到控制台当中
            report = classification_report(y_true, y_pred, digits=4)
            logger.info("\n%s", report)
            output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
            writer = open(output_eval_file, "a")
            print("***** DEV Eval results *****")
            print("\n%s", report)
            writer.write(report)
            writer.close()

            cur_f1 = f1_score(y_true, y_pred)
            # 保存最优模型
            if cur_f1 > best_f1_score:
                print("Saving model ...")
                best_f1_score = cur_f1
                writer = open(output_eval_file, "a")
                writer.write("Model Saved\n")
                writer.close()

                # 应该是
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)

                # 记住这里的示例代码
                torch.save(model_to_save.state_dict(), output_model_file)

                # 保存配置文件
                output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
                with open(output_config_file, 'w') as f:
                    f.write(model_to_save.config.to_json_string())

                label_map = {i: label for i, label in enumerate(label_list, 1)}
                model_config = {"bert_model": args.bert_model, "do_lower": args.do_lower_case,
                                "max_seq_length": args.max_seq_length, "num_labels": len(label_list) + 1,
                                "label_map": label_map}
                json.dump(model_config, open(os.path.join(args.output_dir, "model_config.json"), "w"))
                print("Model Saved")
                best_round = _
            else:
                if _ - best_round > args.early_stop:
                    print("Early Stop!")
                    break

        # Load a trained model and config that you have fine-tuned
    else:
        # 如果不训练的话，就直接将最优模型给加载到内存中来
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        config = BertConfig(output_config_file)
        model = Ner(config, num_labels=num_labels)
        model.load_state_dict(torch.load(output_model_file))

    model.to(device)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):

        out_file = open(args.pred_file, "w", encoding='utf-8')
        eval_examples = processor.get_test_examples(args.test_file)
        eval_features, raw_data = convert_examples_to_features(eval_examples, label_list, args.max_seq_length,
                                                               tokenizer)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_valid_ids = torch.tensor([f.valid_ids for f in eval_features], dtype=torch.long)
        all_lmask_ids = torch.tensor([f.label_mask for f in eval_features], dtype=torch.long)
        all_fea_ids = torch.tensor([f.fea_ids for f in eval_features], dtype=torch.long)
        all_raw_datas = torch.tensor([f.raw_data for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_valid_ids,
                                  all_lmask_ids, all_fea_ids, all_raw_datas)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        y_true = []
        y_pred = []
        label_map = {i: label for i, label in enumerate(label_list, 1)}
        for input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask, fea_ids, tid in tqdm(eval_dataloader,
                                                                                                   desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            valid_ids = valid_ids.to(device)
            label_ids = label_ids.to(device)
            l_mask = l_mask.to(device)
            fea_ids = fea_ids.to(device)
            tid = tid.to(device)

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask, valid_ids=valid_ids, attention_mask_label=l_mask,
                               fea_ids=fea_ids)

            logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            input_mask = input_mask.to('cpu').numpy()

            for i, label in enumerate(label_ids):
                temp_1 = []
                temp_2 = []
                for j, m in enumerate(label):
                    if j == 0:
                        continue
                    elif label_ids[i][j] == 11:
                        y_true.append(temp_1)
                        y_pred.append(temp_2)
                        # assert(len(temp_2) == len(raw_data[tid[i][0]]))
                        SS = []
                        for k in range(len(temp_2)):
                            tmp2k = temp_2[k]
                            if tmp2k != "O" and "-" not in tmp2k:
                                tmp2k = "O"
                            SS.append(raw_data[tid[i][0]][k] + " " + tmp2k)
                        out_file.write("\n".join(SS))
                        out_file.write("\n\n")
                        break
                    else:
                        tmp_label = label_map.get(label_ids[i][j], "O")
                        # if "MISC" in tmp_label:
                        #     tmp_label = "O"
                        temp_1.append(tmp_label)

                        tmp_label = label_map.get(logits[i][j], "O")
                        # if "MISC" in tmp_label:
                        #     tmp_label = "O"
                        temp_2.append(tmp_label)

        report = classification_report(y_true, y_pred, digits=4)
        logger.info("\n%s", report)
        output_eval_file = os.path.join(args.output_dir, "eval_test_results.txt")
        with open(output_eval_file, "a") as writer:
            logger.info("***** Eval Test results *****")
            logger.info("\n%s", report)
            writer.write(report)


if __name__ == "__main__":
    main()
