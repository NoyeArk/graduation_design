import yaml

from train import Pipeline
from data_utils import Data
from meta_data import MetaData
from model.seq_learn import SeqLearn


if __name__ == "__main__":
    with open("D:/Code/graduation_design/rec/config/seq_llm.yaml", 'r', encoding='utf-8') as f:
        config = yaml.unsafe_load(f)
    print(config)

    data = Data(config['data'], 0)
    meta_data = MetaData(config['meta'], data)
    model = SeqLearn(config['model'], data)
    pipeline = Pipeline(config['train'], data, meta_data, model)
    pipeline.train()


"""
# 模型参数设置
seed = 0
factor = 128

# 不同数据集的批次大小设置
batch_size = {
    'Amazon_App': 1024,
    'Kindle': 1024, 
    'Clothing': 2048,
    'Grocery': 2048,
    'Instant_Video': 256,
    'Games': 1024,
    'ml-1m': 1024
}

# 不同数据集的权重系数设置
tradeoff = {
    'Amazon_App': 1,
    'Clothing': 2,
    'Grocery': 128,
    'Kindle': 128,
    'Instant_Video': 2,
    'Games': 32,
    'ml-1m': 2
}

# 不同数据集的训练轮数设置
epoch = {
    'Amazon_App': 15,
    'Kindle': 5,
    'Clothing': 10, 
    'Grocery': 20,
    'Instant_Video': 20,
    'Games': 10,
    'ml-1m': 10
}

# 不同数据集的最大序列长度设置
maxlen = {
    'Amazon_App': 5,
    'Kindle': 3,
    'Clothing': 3,
    'Grocery': 5,
    'Instant_Video': 5,
    'Games': 5,
    'ml-1m': 5
}

# 提出方法及其消融实验的设置说明
# method_name: tradeoff, user_module, model_module, div_module
# SEM: tradeoff[data], 'SAtt', 'dynamic', 'cov'
# w/o uDC: tradeoff[data], 'static', 'dynamic', 'cov'
# w/o bDE: tradeoff[data], 'SAtt', 'static', 'cov'
# w/o Div: 0.0, 'SAtt', 'dynamic', 'cov'
# w/o TPDiv: tradeoff[data], 'SAtt', 'dynamic', 'AEM-cov'
"""