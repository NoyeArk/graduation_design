from model.aem import AEM
from model.ensrec import EnsRec
from model.stream import StackingModel


def get_model(model_type, args, data_args, n_user, n_item):
    if model_type == 'ensrec':
        return EnsRec(args, data_args, n_user, n_item)
    elif model_type == 'aem':
        return AEM(args, data_args, n_user, n_item)
    elif model_type == 'stack':
        return StackingModel(args, data_args, n_user)
    else:
        raise ValueError(f'不支持的模型类型: {model_type}')
