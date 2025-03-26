from model.ensrec import EnsRec


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
    if model_type == 'ensrec':
        return EnsRec(args, data_args, n_user, n_item)
    else:
        raise ValueError(f'不支持的模型类型: {model_type}')
