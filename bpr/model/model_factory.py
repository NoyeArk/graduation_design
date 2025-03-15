from model.sem import Sem
from model.bpr_rec import BPRSeqLearn
from model.rmse_rec import RMSESeqLearn
from model.softmax_rec import SoftmaxSeqLearn


def get_model(model_type, args, data_args, n_user, n_item):
    """
    根据模型类型返回对应的模型实例
    
    Args:
        model_type: 模型类型,可选 'BPR' 或 'RMSE'
        args: 模型参数
        data_args: 数据参数
        n_user: 用户数量
        n_item: 物品数量

    Returns:
        对应类型的模型实例
    """
    if model_type == 'BPR':
        return BPRSeqLearn(args, data_args, n_user, n_item)
    elif model_type == 'RMSE':
        return RMSESeqLearn(args, data_args, n_user, n_item)
    elif model_type== 'SEM':
        return Sem(args, data_args, n_user, n_item)
    elif model_type == 'softmax':
        return SoftmaxSeqLearn(args, data_args, n_user, n_item)
    else:
        raise ValueError(f'不支持的模型类型: {model_type}')
