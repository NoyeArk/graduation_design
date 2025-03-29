import yaml

from data_process import Data
from model.acf import AcfTrain
from model.pfmc import PfmcTrain
from model.fdsa import FdsaTrain
from model.anam import AnamTrain
from model.harnn import HarnnTrain
from model.caser import CaserTrain
from model.sasrec import SasrecTrain


if __name__ == '__main__':
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    data = Data(config['dataset'], config['seed'])

    model_mapping = {
        'acf': AcfTrain,
        'pfmc': PfmcTrain,
        'fdsa': FdsaTrain,
        'caser': CaserTrain,
        'harnn': HarnnTrain,
        'anam': AnamTrain,
        'sasrec': SasrecTrain
    }

    pipeline = model_mapping[config['model']](config, data)
    pipeline.train(use_item_attributes=False)
