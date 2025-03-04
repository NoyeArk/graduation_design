import copy
import toolz
import argparse
import numpy as np
from tqdm import tqdm

from utils import *
from GCNdata import Data
from meta_data import MetaData
from llm_rec.seq_llm import Llm4SeqRec


def parse_args(name, factor, batch_size, tradeoff, user_module, model_module, div_module, epoch, maxlen):
    parser = argparse.ArgumentParser(description="Run .")
    parser.add_argument('--name', nargs='?', default= name)
    parser.add_argument('--model', nargs='?', default='SASEM')
    parser.add_argument('--path', nargs='?', default='D:/Code/graduation_design/data/'+name,
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default=name,
                        help='Choose a dataset.')
    parser.add_argument('--batch_size', type=int, default=batch_size,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=factor,
                        help='Number of hidden factors.')
    parser.add_argument('--lamda', type=float, default = 0.00001,
                        help='Regularizer for bilinear part.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')
    parser.add_argument('--epoch', type=int, default=epoch)
    parser.add_argument('--tradeoff', type=float, default=tradeoff)
    parser.add_argument('--user_module', nargs='?', default=user_module)
    parser.add_argument('--model_module', nargs='?', default=model_module)
    parser.add_argument('--div_module', nargs='?', default=div_module)
    parser.add_argument('--maxlen', type=int, default=maxlen)
    parser.add_argument('--optimizer', nargs='?', default='AdamOptimizer',
                        help='Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer).')
    return parser.parse_args()


class Pipeline(object):
    def __init__(self, args, data, meta_data):
        self.args = args
        self.epoch = self.args.epoch
        self.batch_size = args.batch_size
        self.seq_max_len = args.maxlen
        self.print_train = False
        # Data loading
        self.data = data
        self.meta_data = meta_data
        self.entity = self.data.entity
        self.n_user = self.data.entity_num['user']
        self.n_item = self.data.entity_num['item']
        # Training\\\建立模型
        self.model = Llm4SeqRec(
            self.args,
            self.data,
            args.hidden_factor,
            args.lr,
            args.lamda,
            args.optimizer
        )
        with open("./process_result.txt", "a") as f:
             f.write("dataset:%s\n" % (args.name))

    def train(self):
        # 初始结果
        MAP_valid = 0
        p = 0

        max_map, max_ndcg, max_prec = (0, 0), (0, 0), (0, 0)
        for epoch in range(0, self.epoch + 1):  # 每一次迭代训练
            shuffle = np.arange(len(self.meta_data.user_item_pairs))
            # np.random.shuffle(shuffle)

            # 用户-物品对
            user_item_pairs = self.meta_data.user_item_pairs  # none * 2

            self.users = user_item_pairs[:, 0]
            self.times = self.timestamp()
            self.items = user_item_pairs[:, 1]

            # 采样，none * NG
            negative_sample_count = 1
            self.negative_samples = self.sample_negative(
                user_item_pairs=user_item_pairs,
                meta_data=self.meta_data.train_meta,
                negative_sample_count=negative_sample_count
            )

            # 序列 none * seq
            self.seq = np.array([self.data.latest_interaction[(line[0], line[1])] for line in user_item_pairs])

            # 正样本标签
            meta_positive = self.meta_data.user_item_pairs_labels #none * k  k denotes BM number

            # 负样本标签
            meta_negative = self.meta_data.label_negative(
                self.negative_samples,
                negative_sample_count
            )  # none * NG * k

            # 基模型训练
            base_focus = self.meta_data.train_meta[:, :, 2: 2 + self.seq_max_len] #none * k * p # p denotes window size

            # 批量训练
            for user_chunk in tqdm(toolz.partition_all(self.batch_size, [i for i in range(len(user_item_pairs))])):
                p = p + 1
                chunk = shuffle[list(user_chunk)]

                u_chunk = self.users[chunk]  # none
                seq_chunk = self.seq[chunk]  # none * p
                i_pos_chunk = self.items[chunk]  # none
                i_neg_chunk = self.negative_samples[chunk]  # none * NG

                # 正负样本标签
                meta_positive_chunk = meta_positive[chunk]  # none * k
                meta_negative_chunck = meta_negative[chunk]  # none * NG * k

                # 基模型表示
                base_focus_chunck = base_focus[chunk]
                times = self.times[chunk]

                self.feed_dict = {
                    'u': u_chunk,
                    'seq': seq_chunk,
                    'i_pos': i_pos_chunk,
                    'i_neg': i_neg_chunk,
                    'meta_pos': meta_positive_chunk,
                    'meta_neg': meta_negative_chunck,
                    'base_focus': base_focus_chunck,
                    'times': times
                }
                loss = self.model.partial_fit(self.feed_dict)

            # 评估训练和验证数据集
            if epoch % 1 == 0:
                print(f"Loss {loss[0]:.4f}\t{loss[1]:.4f}")

                # 评估训练和验证数据集
                if self.print_train:
                    init_test_TopK_train = self.evaluate_TopK(
                        test=self.data.valid_set[:10000],
                        test_meta=self.meta_data.train_meta[:10000],
                        topk=[10]
                    )
                    print(init_test_TopK_train)

                # 评估测试集
                init_test_TopK_test = self.evaluate_TopK(
                    test=self.data.test_set,
                    test_meta=self.meta_data.test_meta,
                    topk=[20, 50]
                )

                print(f"Epoch {epoch} \t TEST SET MAP: {init_test_TopK_test[0]:.4f}, NDCG: {init_test_TopK_test[1]:.4f}, PREC: {init_test_TopK_test[2]:.4f}\n")

                with open("./process_result.txt", "a") as f:
                    f.write(f"Epoch {epoch} \t TEST SET MAP: {init_test_TopK_test[0]:.4f}, NDCG: {init_test_TopK_test[1]:.4f}, PREC: {init_test_TopK_test[2]:.4f}\n")

                if MAP_valid < np.mean(init_test_TopK_test[4:]):
                    MAP_valid = np.mean(init_test_TopK_test[4:])
                    result_print = init_test_TopK_test

                max_map = (max(max_map[0], result_print[0]), max(max_map[1], result_print[3]))
                max_ndcg = (max(max_ndcg[0], result_print[1]), max(max_ndcg[1], result_print[4]))
                max_prec = (max(max_prec[0], result_print[2]), max(max_prec[1], result_print[5]))

        # 保存最终模型
        # self.model.save_model(f"./models/{self.args.name}_{self.args.model}/.ckpt")

        with open("./result.txt","a") as f:
            f.write("{},{},{},{},{},{},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f}\n".format(
                self.args.name,
                self.args.model,
                self.args.user_module,
                self.args.model_module,
                self.args.div_module,
                self.args.tradeoff,
                max_map[0],
                max_ndcg[0],
                max_prec[0],
                max_map[1],
                max_ndcg[1],
                max_prec[1]
                # result_print[0],
                # result_print[1],
                # result_print[2],
                # result_print[3],
                # result_print[4],
                # result_print[5]
            ))

    def sample_negative(self, user_item_pairs, meta_data, negative_sample_count):
        """
        采样负样本的函数

        Args:
            user_item_pairs (`np.ndarray`): 用户-物品对
            meta_data (`np.ndarray`): 元数据
            negative_sample_count (`int`): 负样本数量

        返回值:
            样本 (`np.ndarray`): 负样本
        """
        # 基础模型训练
        BM_train = self.data.dict_forward['train']

        # 只使用每个基础模型预测的前 50 个排名结果
        top_k_results = 50
        meta_data = meta_data[:, :, 2:2+top_k_results]

        # 重塑元数据
        meta_data = np.reshape(meta_data, [len(meta_data), -1])  # [none, k*50]
        num_rows, _ = meta_data.shape
        sample = []

        # 进行 negative_sample_count 次负样本生成
        for _ in range(negative_sample_count):
            # 随机生成样本
            sample_i = np.random.randint(0, self.n_item, num_rows)
            for j, item in enumerate(sample_i):
                # 如果样本在基础模型中，则重新生成样本
                if item in BM_train[user_item_pairs[j, 0]]:
                    sample_i[j] = np.random.randint(0, self.n_item)
            sample.append(sample_i)

        return np.stack(sample, axis=-1)

    def timestamp(self):
        """
        返回时间戳的函数
        """
        # 初始化时间戳
        t = np.ones(len(self.users))

        # 初始化计数器
        s, cout, c = self.users[0], 10, 1
        for i, ur in enumerate(self.users):
            if ur == s:
                cout += 1
                c += 1
                t[i] = cout
            else:
                t[i - c: i] = t[i - c: i] / cout
                cout = 10
                c = 1
                s = ur
                t[i] = cout
        t[i - c + 1:] = t[i - c + 1:] / cout
        return t

    def evaluate_TopK(self, test, test_meta, topk):
        """
        评估 TopK 的函数。

        Args:
            test (`np.ndarray`): 测试数据
            test_meta (`np.ndarray`): 测试元数据
            topk (`list`): TopK 列表

        Returns:
            result_MAP (`dict`): MAP 结果
            result_PREC (`dict`): PREC 结果
            result_NDCG (`dict`): NDCG 结果
        """
        # 获取测试数据
        user_item_pairs = copy.deepcopy(np.array(test))  # none * 2
        size = len(user_item_pairs)

        # 初始化结果字典
        result_map = {key: [] for key in topk}
        result_ndcg = {key: [] for key in topk}
        result_recall = {key: [] for key in topk}

        # 初始化每一个用户-物品对的最后一次交互
        last_iteraction = []  # none * 5
        for line in user_item_pairs:
            user, item = line
            last_iteraction.append(self.data.latest_interaction[(user, item)])

        # 将最后一次交互转换为数组
        last_iteraction = np.array(last_iteraction)

        # 分块处理
        num = 999  # self.n_user

        for i in range(int(size / num + 1)):
            beg, end = i * num, (i + 1) * num
            user_item_pairs_block = user_item_pairs[beg: end]
            last_iteraction_block = last_iteraction[beg: end]
            items_score = self.meta_data.all_score(test_meta[beg: end])
            base_focus = test_meta[beg:end, :, 2:2 + self.seq_max_len]

            # 预测得分
            pred_items, wgts = self.model.topk(
                user_item_pairs=user_item_pairs_block,
                last_interaction=last_iteraction_block,
                items_score=items_score,
                base_focus=base_focus
            )  # none * 50
            assert len(pred_items) == len(user_item_pairs_block)

            # 评估
            for i, (user, gt_item) in enumerate(user_item_pairs_block):
                # 对于每一个用户-物品对，计算 topk 的指标
                for top_n in topk:
                    useful_item_cnt = 0
                    for pred_item in pred_items[i]:
                        if useful_item_cnt == top_n:
                            result_map[top_n].append(0.0)
                            result_ndcg[top_n].append(0.0)
                            result_recall[top_n].append(0.0)
                            useful_item_cnt = 0
                            break
                        elif pred_item == gt_item:
                            result_recall[top_n].append(1.0)
                            result_ndcg[top_n].append(np.log(2) / np.log(useful_item_cnt + 2))
                            result_map[top_n].append(1 / (useful_item_cnt + 1))
                            useful_item_cnt = 0
                            break
                        elif pred_item in (self.data.set_forward['train'][user] or self.data.set_forward['valid'][user]):
                            continue
                        else:
                            useful_item_cnt += 1
        return [
            np.mean(result_map[topk[0]]),
            np.mean(result_ndcg[topk[0]]),
            np.mean(result_recall[topk[0]]),
            np.mean(result_map[topk[1]]),
            np.mean(result_ndcg[topk[1]]),
            np.mean(result_recall[topk[1]])
        ]


def seq_llm_main(
    name,
    factor,
    batch_size,
    tradeoff,
    user_module,
    model_module,
    div_module,
    epoch,
    maxlen
):
    """
    主函数

    Args:
        name (`str`): 数据集名称
        factor (`int`): 隐向量维度
        batch_size (`int`): 批量大小
        tradeoff (`float`): 权衡参数
        user_module (`str`): 用户模块
        model_module (`str`): 模型模块
        div_module (`str`): 多样性模块
        epoch (`int`): 训练轮数
        maxlen (`int`): 最大序列长度
    """
    args = parse_args(
        name,
        factor,
        batch_size,
        tradeoff,
        user_module,
        model_module,
        div_module,
        epoch,
        maxlen
    )
    print(args)
    data = Data(args, 0)  # 获取数据
    meta_data = MetaData(args, data)
    pipeline = Pipeline(args, data, meta_data)
    pipeline.train()
