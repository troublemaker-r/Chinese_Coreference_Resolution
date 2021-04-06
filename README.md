# SpanBERT for Chinese Coreference Resolution (Pytorch)

- 参考论文： [SpanBERT: Improving Pre-training by Representing and Predicting Spans](https://arxiv.org/abs/1907.10529)
- 参考开源代码：[https://github.com/mandarjoshi90/coref](https://github.com/mandarjoshi90/coref)
  - 上述参考代码使用`tensorflow`实现，并使用英文数据集
- 预训练模型下载地址：  
  - 中文预训练`RoBERTa`模型 [https://github.com/brightmart/roberta_zh](https://github.com/brightmart/roberta_zh)
  - 中文预训练`BERT-wwm`模型 [https://github.com/brightmart/roberta_zh](https://github.com/brightmart/roberta_zh)
  -  中文预训练`Bert`模型 [https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz)

#### :ledger: 1. 代码架构：

~~~text
│ conll.py
│ coreference.py
│ demo.py
│ metrics.py
│ utils.py
│ experiments.conf
│ requirements.txt
│
├─bert
│ │ modeling.py
│ │ optimization.py
│ │ tokenization.py
│ 
├─conll-2012
│ └─scorer
│ ├─reference-coreference-scorers
│ └─v8.01
├─data
│ ├─dev 
│ ├─test 
│ └─train
│ 
└─pretrain_model
 │ bert_config.json
 │ pytorch_model.bin 
 │ vocab.txt
~~~

**其中**：

**conll.py**：验证集验证所需脚本

**coreference.py**：指代消解模型脚本

**demo.py**：指代消解工程测试脚本

**metrics.py**：验证集计算指标脚本

**utils.py**：数据转换，文件读写脚本

**experiments.conf**：代码运行所需参数配置文件

**requirements.txt**：代码运行必要环境文件

**bert**：用于存放bert模型相关脚本文件 

**conll-2012**：官方提供的验证文件

**data**：用于存放训练验证预测文件以及最后预测的结果文件

**pretrain_model**：用于存放预训练模型（包含模型、参数配置文件、字典）

#### :orange_book: 2. 运行环境

- 运行环境要求python版本在3.5及以上，运行环境配置见`requirements.txt`
- 一块`TITAN xp` , 参数选择`[ffnn_size=2000，nun_epochs=30]`，需要7小时左右

#### :green_book: ​3. 运行方式

- 在`experiments.conf`文件中配置好向相应的参数，命令行运行：`python demo.py`
  即可，默认使用第三块GPU(编号为2)。
