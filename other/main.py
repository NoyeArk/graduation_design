import yaml
from other.train import Pipeline


if __name__ == "__main__":
    # 读取配置文件
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # 从配置文件中获取参数
    pipeline = Pipeline(
        embedding_dim=config['model']['embedding_dim'],
        num_next_items=config['model']['num_next_items'],
        batch_size=config['train']['batch_size'],
        epochs=config['train']['epochs'],
        lr=config['train']['lr']
    )

    print("开始训练模型...")
    pipeline.train()

    print("\n开始评估模型...")
    ndcg = pipeline.evaluate()
    print(f"模型 NDCG: {ndcg:.4f}")
