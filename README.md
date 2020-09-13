# sequence-labeling-models  
[![Join the chat at https://gitter.im/bupt-nlp/sequence-labeling-models](https://badges.gitter.im/bupt-nlp/sequence-labeling-models.svg)](https://gitter.im/bupt-nlp/sequence-labeling-models?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

we will share our research model at here. 

## All of things you should do

- Install

```shell
pip install -r requirements.txt
```

- Process your data to the file format

> this is the sequence labeling task model, so if you process your data to the file format, you can fed the file to our model at free. And the model will return the specific format file to you. The following is the format of the input file

```shell
token1 token2 token3 ... tokenn\tlabel1 label2 label3 ... labeln
```

- Complete configuration

```shell
python main.py \
    --do_train=True \
    --do_eval=True \
    --do_predict=True \
    --input_file=data.train \
    --output_file=data.train.result \
    --batch_size=10 \
    --lr=5e-5 \
    --hidden_size=768 \
```

## BiLSTM CRF

![](./bilstm_crf/imgs/model.png)


## Bert CRF

![](./bert_crf/imgs/bert_crf.png)

## Bert + External Features + CRF

- Bert 层

Bert 在模型当中用作一个编码层，主要负责将核心语义信息添加到进行编码，然后输出一个具备丰富上下文语义的隐藏层特征向量。

- External 层

可以融合外部知识，将`POS`，`NER`信息融入到序列标注的任务当中，然后将其进行深度融合。这些信息融合的方式也是可以对其进行深入检索。

- CRF 层

作为一个目前常用且高效的序列信息解码层，能够在全局获得最佳预测效果。
