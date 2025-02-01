from SeqEnsemble import SEM_main

# 模型参数设置
factor = 32
seed = 0

# 不同数据集的批次大小设置
batch_size = {
    'Amazon_App': 1024,
    'Kindle': 1024, 
    'Clothing': 2048,
    'Grocery': 2048,
    'Instant_Video': 256,
    'Games': 1024
}

# 不同数据集的权重系数设置
tradeoff = {
    'Amazon_App': 1,
    'Clothing': 2,
    'Grocery': 128,
    'Kindle': 128,
    'Instant_Video': 2,
    'Games': 32
}

# 不同数据集的训练轮数设置
epoch = {
    'Amazon_App': 15,
    'Kindle': 5,
    'Clothing': 10, 
    'Grocery': 20,
    'Instant_Video': 20,
    'Games': 10
}

# 不同数据集的最大序列长度设置
maxlen = {
    'Amazon_App': 5,
    'Kindle': 3,
    'Clothing': 3,
    'Grocery': 5,
    'Instant_Video': 5,
    'Games': 5
}

# 提出方法及其消融实验的设置说明
# method_name: tradeoff, user_module, model_module, div_module
# SEM: tradeoff[data], 'SAtt', 'dynamic', 'cov'
# w/o uDC: tradeoff[data], 'static', 'dynamic', 'cov'
# w/o bDE: tradeoff[data], 'SAtt', 'static', 'cov'
# w/o Div: 0.0, 'SAtt', 'dynamic', 'cov'
# w/o TPDiv: tradeoff[data], 'SAtt', 'dynamic', 'AEM-cov'

# 运行示例
data = 'Amazon_App'

SEM_main(
    name=data,
    factor=factor,
    batch_size=batch_size[data],
    tradeoff=tradeoff[data],
    user_module='SAtt',
    model_module='dynamic',
    div_module='cov',
    epoch=epoch[data],
    maxlen=maxlen[data]
)
