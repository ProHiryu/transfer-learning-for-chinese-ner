# transfer-learning-for-chinese-ner

> follow paper TRANSFER LEARNING FOR SEQUENCE TAGGING WITH HIERARCHICAL RECURRENT NETWORKS

## Models:

![](http://ww1.sinaimg.cn/large/e1ac6bd5ly1fwq2lqapizj21ba16ajzt.jpg)

本文将transfer learning分为了三个层次

1. cross-domain transfer where label mapping is possible
2. cross-domain transfer with disparate label sets
3. cross-lingual transfer

这里仅先实现第二点，作中文NER，原文GitHub是theano实现的，这里用tensorflow

## Requirements

```
tensorflow > 1.10
python == 3.65
```