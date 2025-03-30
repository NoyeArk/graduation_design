import os
import copy
import toolz
import numpy as np
from tqdm import tqdm
from time import time

include_valid = True


class Pipeline(object):
    """
    基础训练类
    """
    def __init__(self, args, data):
        self.args = args
        self.data = data
        self.batch_size = args['train']['batch_size']
        self.epoch = args['train']['epoch']
        self.entity = self.data.entity
        self.user_side_entity = self.data.user_side_entity
        self.item_side_entity = self.data.item_side_entity
        # Data loading
        self.n_user = self.data.entity_num['user']
        self.n_item = self.data.entity_num['item']
        self.include_valid = include_valid

    def handle_loss(self, epoch, loss):
        """
        处理损失

        Args:
            epoch (`int`): 当前训练轮数
            loss (`float`): 损失值
        """
        print(f"Epoch {epoch} - Loss: {loss:.4f}")

    def train(self, use_item_attributes=False):
        """
        训练模型

        Args:
            use_item_attributes (`bool`): 是否使用物品属性
        """
        self.use_item_attributes = use_item_attributes

        best_metric = 0
        if include_valid:
            positive_samples = np.array(self.data.train_set)  # Array 形式的二元组(user, item), none * 2
            basemodel = 'basemodel_v'
        else:
            basemodel = 'basemodel'

        # 这里是用于训练属性模型的数据准备部分
        positive_samples_concat_last5 = np.concatenate(
            [
                positive_samples,
                np.array([self.data.latest_interaction.get((line[0], line[1]), [0] * 5)
                           for line in positive_samples])
            ],
            axis=1
        )  # none * 2 + 5

        for epoch in tqdm(range(0, self.epoch + 1)): #每一次迭代训练
            np.random.shuffle(positive_samples)
            # sample 负样本采样
            negative_sample_num = 1  # NG 倍举例
            negative_samples = self.sample_negative(
                positive_samples,
                negative_sample_num
            )  # 采样, none * NG

            for user_chunk in toolz.partition_all(self.batch_size, [i for i in range(len(positive_samples))]):
                chunk = list(user_chunk)

                negative_samples_chunk = np.array(negative_samples[chunk], dtype=np.int64)  # none * 1
                positive_samples_concat_last5_trunk = positive_samples_concat_last5[chunk]  # none * 2 + 5
                positive_samples_concat_last5_trunk_copy = copy.deepcopy(positive_samples_concat_last5_trunk)
                positive_samples_concat_last5_trunk_copy[:, 1] = negative_samples_chunk

                feedback = np.stack([positive_samples_concat_last5_trunk, positive_samples_concat_last5_trunk_copy], axis=1)
                feedback = np.reshape(feedback,[-1, 2 + 5])
                labels = np.reshape(
                    np.stack([np.ones(len(chunk)), np.zeros(len(chunk))], axis=1),
                    [-1, 1]
                )

                # 模型输入特征
                if use_item_attributes:
                    self.feed_dict = {
                        'feedback': feedback,
                        'labels': labels,
                        'item_attributes': self.item_attributes
                    }
                else:
                    self.feed_dict = {
                        'feedback': feedback,
                        'labels': labels
                    }
                loss = self.model.partial_fit(self.feed_dict)

            t2 = time()

            # evaluate training and validation datasets
            if epoch % int(self.args['train']['epoch'] / 10) == 0:
                for topk in [10]:
                    test_result = self.evaluate_topk(self.data.test_set, topk)
                    print(
                        f"Epoch {epoch} Top{topk} \t TEST SET:0.0000 MAP:{test_result[0]:.4f}, "
                        f"NDCG:{test_result[1]:.4f}, PREC:{test_result[2]:.4f} [{time() - t2:.1f}s]\n"
                    )

                if best_metric <= np.sum(test_result) and epoch < self.epoch:
                    best_metric = np.sum(test_result)
                    meta_result = self.save_meta_result()

        dir_name = f"../base_model_results/{self.args['dataset']['name']}"
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        np.save(
            f"../base_model_results/{self.args['dataset']['name']}/{self.args['model']}.npy", 
            meta_result
        )

    def save_meta_result(self):
        """
        保存 meta-path 特征

        Returns:
            `np.ndarray`: 保存的结果
        """
        user_item_pairs = copy.deepcopy(self.data.dict_list['user_item'])
        size = len(user_item_pairs)
        num = 200

        last_iteraction = []
        for line in user_item_pairs:
            user, item = line
            last_iteraction.append(self.data.latest_interaction[(user, item)])
        last_iteraction = np.array(last_iteraction)

        # 计算 topk 得分
        score = []
        for i in range(int(size / num + 1)):
            user_item_block = user_item_pairs[i*num: (i+1)*num]
            last_iteraction_block = last_iteraction[i*num: (i+1)*num]
            feedback_block = np.concatenate((user_item_block, last_iteraction_block), axis=1)
            try:
                score_block = self.model.topk(feedback_block, self.item_attributes)
            except:
                score_block = self.model.topk(feedback_block)

            score_block = score_block[:, :100].tolist()
            score.extend(score_block)

        score = np.array(score,dtype=np.int64)
        return np.concatenate((user_item_pairs, score), axis=1)

    def sample_negative(self, positive_samples, sample_num=10):
        """
        采样负样本

        Args:
            positive_samples (`np.ndarray`): 二元组(user, item), none * 2
            sample_num (`int`): 采样数目

        Returns:
            `np.ndarray`: 采样的负样本
        """
        samples = np.random.randint(0, self.n_item, size=(len(positive_samples), sample_num))
        return samples

    def collect_attributes(self):
        """
        收集属性
        """
        NUM = 3
        attributes = []
        start_index = 0
        for entity in self.data.item_side_entity:
            key = 'item_' + entity
            attribute_item = self.data.dict_forward[key]
            attribute = []
            for item in range(self.n_item):
                list_ = attribute_item[item]
                if len(list_) <= NUM:
                    attribute.append(list_+[-1 for i in range(NUM-len(list_))])
                else:
                    attribute.append(list_[:NUM])
            attribute = np.array(attribute)
            attributes.append(attribute + start_index)
            start_index = start_index + self.data.entity_num[entity]
        return np.stack(attributes, axis=1)

    def evaluate_topk(self, test_set, topk):
        """
        评估TopK物品

        Args:
            test_set (`list`): 测试集，包含用户和物品的二元组
            topk (`int`): 评估的TopK值

        Returns:
            `list`: 评估结果, 包括MAP、NDCG和PREC的平均值
        """
        test_candidate = copy.deepcopy(np.array(test_set))  # none * 2
        size = len(test_candidate)
        result_map = []
        result_recall = []
        result_ndcg = []

        num = 100
        # meta-path feature
        last_iteraction = []  # none * 5
        for line in test_candidate:
            #meta-path特征
            user,item = line
            last_iteraction.append(self.data.latest_interaction[(user,item)])

        last_iteraction = np.array(last_iteraction)
        for _ in range(int(size/num+1)):
            user_item_block = test_candidate[_*num:(_+1)*num]
            last_iteraction_block = last_iteraction[_*num:(_+1)*num]
            feedback_block = np.concatenate((user_item_block,last_iteraction_block),axis=1)
            try:
                prediction = self.model.topk(feedback_block, self.item_attributes) #none * 50
            except:
                prediction = self.model.topk(feedback_block) #none * 50

            assert len(prediction) == len(feedback_block)
            for i, line in enumerate(user_item_block):
                user, item = line
                n = 0
#                print(prediction[i])
                for it in prediction[i]:
                    if n > topk - 1:
                        result_map.append(0.0)
                        result_ndcg.append(0.0)
                        result_recall.append(0.0)
                        n = 0
                        break
                    elif it == item:
                        # print([it,item])
                        result_map.append(1.0)
                        result_ndcg.append(np.log(2) / np.log(n + 2))
                        result_recall.append(1 / (n + 1))
                        n = 0
                        break
                    elif it in self.data.set_forward['train'][user] or it in self.data.set_forward['valid'][user]:
                        continue
                    else:
                        n = n + 1

        print(np.sum(result_map))
        return [np.mean(result_map), np.mean(result_ndcg), np.mean(result_recall)]
