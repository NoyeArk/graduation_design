import torch
import torch.nn as nn
from tqdm import tqdm

from data_process import Data
from model.baseline import NextKItemPredictor


class Pipeline:
    """
    训练模型的流水线类
    """
    def __init__(self, embedding_dim, num_next_items, batch_size, epochs, lr):
        """
        初始化流水线参数
        
        Args:
            embedding_dim (int): 嵌入维度
            num_next_items (int): 预测接下来的物品数量
            batch_size (int): 批次大小
            epochs (int): 训练轮数
            lr (float): 学习率
        """
        self.data = Data()

        self.model = NextKItemPredictor(
            num_users=self.data.num_users,
            num_items=self.data.num_movies,
            embedding_dim=embedding_dim,
            num_next_items=num_next_items
        )

        self.batch_size = batch_size
        self.epochs = epochs
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self):
        """
        训练模型
        """
        for epoch in tqdm(range(self.epochs), desc='训练进度'):
            total_loss = 0
            batch_count = 0

            # 遍历每个用户的训练数据
            self.model.train()
            for user_id, movie_list in self.data.train_data.items():
                if len(movie_list) < 2:
                    continue

                # 准备训练数据
                user_tensor = torch.LongTensor([user_id])
                history_tensor = torch.LongTensor([movie_list[:-1]])
                target_tensor = torch.zeros(1, self.data.num_movies)
                target_tensor[0, movie_list[-1]] = 1

                # 前向传播
                pred = self.model(user_tensor, history_tensor)
                loss = self.criterion(pred, target_tensor)

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                batch_count += 1

                if batch_count % self.batch_size == 0:
                    print(f'Epoch {epoch+1}, Batch {batch_count//self.batch_size}, '
                          f'Average Loss: {total_loss/self.batch_size:.4f}')
                    total_loss = 0

            # 评估模型
            ndcg = self.evaluate()
            print(f'Epoch {epoch+1}, NDCG: {ndcg:.4f}')

    def evaluate(self):
        """
        评估模型
        
        Returns:
            float: NDCG分数
        """
        self.model.eval()
        ndcg_scores = []

        with torch.no_grad():
            for user_id, movie_list in self.data.valid_data.items():
                if len(movie_list) < 2:
                    continue

                user_tensor = torch.LongTensor([user_id])
                history_tensor = torch.LongTensor([movie_list[:-1]])

                # 获取推荐列表
                recommended_items = self.model.recommend_next_k_items(user_tensor, history_tensor)[0]

                # 计算DCG
                dcg = 0
                for i, item in enumerate(recommended_items):
                    if item.item() == movie_list[-1]:
                        # 使用log2(i+2)是因为i从0开始
                        dcg += 1 / torch.log2(torch.tensor(i + 2, dtype=torch.float))
                        break

                # 理想情况下的 DCG (IDCG)
                idcg = 1  # 因为只有一个相关项,且相关性为1

                # 计算 NDCG
                ndcg = dcg / idcg if idcg > 0 else 0
                ndcg_scores.append(ndcg)

        # 返回平均 NDCG
        return sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0
