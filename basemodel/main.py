import yaml
import numpy as np

from model.ACF import ACF
from PFMC import PFMC
from model.FDSA import FDSA
from data_process import Data
from pipeline import Pipeline

#'Kindle'!
# for data in ['Instant_Video', 'Amazon_App', 'Kindle', 'Clothing', 'Games', 'Grocery']:

class Train(Pipeline):
    """
    训练类
    """
    def __init__(self, args, data):
        super(Train,self).__init__(args, data)
        self.item_attributes = self.collect_attributes()

        model_classes = {
            'ACF': ACF,
            'PFMC': PFMC,
            'FDSA': FDSA
        }
        if args['model_name'] in model_classes:
            self.model = model_classes[args['model_name']](
                args,
                data,
                args['train']['factor'],
                args['train']['lr'],
                args['train']['lamda'],
                args['train']['optimizer']
            )

    def sample_negative(self, data, num=10):
        """
        从所有的物品中进行随机采样作为负样本

        Args:
            data (`dict`): 数据
            num (`int`): 采样数量

        Returns:
            samples (`np.ndarray`): 采样结果
        """
        samples = np.random.randint(0, self.n_item, size=len(data))
        return samples


if __name__ == '__main__':
    with open('conig.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    data = Data(config['dataset'], config['seed'])
    pipeline = Train(config, data)
    pipeline.train_attribute()
