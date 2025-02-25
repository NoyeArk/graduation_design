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
            embedding_dim (`int`): 嵌入维度
            num_next_items (`int`): 预测接下来的物品数量
            batch_size (`int`): 批次大小
            epochs (`int`): 训练轮数
            lr (`float`): 学习率
        """
        self.data = Data()

        self.model = NextKItemPredictor(
            num_users=self.data.num_users,
            num_items=self.data.num_items,
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

            self.model.train()
            user_ids = list(self.data.train_data.keys())
            num_batches = len(user_ids) // self.batch_size

            for batch_idx in range(num_batches):
                batch_user_ids = user_ids[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
                batch_loss = 0

                for user_id in batch_user_ids:
                    items_list = self.data.train_data[user_id]
                    if len(items_list) < 2:
                        continue

                    # 生成正负样本对
                    pos_item = items_list[-1]
                    neg_item = self.data.sample_negative_item(user_id)

                    user_tensor = torch.LongTensor([user_id])
                    pos_tensor = torch.LongTensor([pos_item])
                    neg_tensor = torch.LongTensor([neg_item])

                    # 前向传播
                    history_tensor = torch.LongTensor(items_list[:-1])
                    pred_pos = self.model(user_tensor, history_tensor, pos_tensor)
                    pred_neg = self.model(user_tensor, history_tensor, neg_tensor)

                    # 使用 pairwise_loss
                    labels = torch.tensor([1.0])  # 假设标签为1表示正样本对
                    loss = self.pairwise_loss(torch.cat((pred_pos, pred_neg)), labels)

                    batch_loss += loss

                # 反向传播
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

                total_loss += batch_loss.item()

                print(f'Epoch {epoch+1}, Batch {batch_idx+1}/{num_batches}, '
                      f'Average Loss: {batch_loss.item()/self.batch_size:.4f}')

            # 评估模型
            ndcg = self.evaluate()
            print(f'Epoch {epoch+1}, NDCG: {ndcg:.4f}')

        self.model.save_checkpoint('model0223.pth')

    def evaluate(self):
        """
        评估模型

        Returns:
            float: NDCG分数
        """
        self.model.eval()
        ndcg_scores = []

        with torch.no_grad():
            for user_id, items_list in self.data.valid_data.items():
                if len(items_list) < 2:
                    continue

                user_tensor = torch.LongTensor([user_id])
                history_tensor = torch.LongTensor([items_list[:-1]])

                # 获取推荐列表
                recommended_items = self.model.recommend_next_k_items(
                    user_tensor,
                    history_tensor
                )[0]

                # 计算DCG
                dcg = 0
                for i, item in enumerate(recommended_items):
                    if item.item() == items_list[-1]:
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

    def pairwise_loss(self, pred, labels):
        """
        成对损失函数

        Args:
            pred (`torch.Tensor`): 预测数据
            labels (`torch.Tensor`): 标签

        Returns:
            loss (`torch.Tensor`): 损失
        """
        inputx_f = torch.cat((pred[1:], torch.zeros_like(pred[0]).unsqueeze(0)), dim=0)
        loss = -torch.sum(torch.log(torch.sigmoid((pred - inputx_f) * labels)))
        return loss
