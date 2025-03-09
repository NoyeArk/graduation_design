import copy
import toolz
import numpy as np
from tqdm import tqdm


class Pipeline(object):
    def __init__(self, args, data, meta_data, model):
        self.args = args
        self.epoch = self.args['epoch']
        self.batch_size = args['batch_size']
        self.data = data
        self.seq_max_len = self.data.args['maxlen']
        self.print_train = False
        self.meta_data = meta_data
        self.entity = self.data.entity
        self.n_user = self.data.entity_num['user']
        self.n_item = self.data.entity_num['item']
        self.model = model

        # with open("./process_result.txt", "a") as f:
        #      f.write("dataset:%s\n" % (args.name))

    def train(self):
        MAP_valid = 0
        p = 0

        max_map, max_ndcg, max_prec = (0, 0), (0, 0), (0, 0)
        for epoch in range(0, self.epoch + 1):  # 每一次迭代训练
            shuffle = np.arange(len(self.meta_data.user_item_pairs))

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
            base_focus = self.meta_data.train_meta[:, :, 2: 2 + self.seq_max_len]  # none * k * p # p denotes window size

            pbar = tqdm(toolz.partition_all(self.batch_size, range(len(user_item_pairs))), desc=f"Epoch {epoch}")
            for user_chunk in pbar:
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
                loss_rec, loss_div = self.model.partial_fit(self.feed_dict)

                pbar.set_postfix({
                    'loss_rec': f'{loss_rec:.4f}',
                    'loss_div': f'{loss_div:.4f}'
                })

            if self.print_train:
                init_test_TopK_train = self.evaluate_TopK(
                    test=self.data.valid_set[:10000],
                    test_meta=self.meta_data.train_meta[:10000],
                    topk=[10]
                )
                print(init_test_TopK_train)

            maps, ndcgs, recalls = self.evaluate_TopK(
                test=self.data.test_set,
                test_meta=self.meta_data.test_meta,
                topk=[20, 50]
            )

            for topk in [20, 50]:
                print(f"------> epoch {epoch} top{topk} map: {maps[topk]}, ndcg: {ndcgs[topk]}, prec: {recalls[topk]}")

            # with open("./process_result.txt", "a") as f:
            #     f.write(f"Epoch {epoch} \t TEST SET MAP: {test_results[0]:.4f}, NDCG: {test_results[1]:.4f}, PREC: {test_results[2]:.4f}\n")

            # if MAP_valid < np.mean(init_test_TopK_test[4:]):
            #     MAP_valid = np.mean(init_test_TopK_test[4:])
            #     result_print = init_test_TopK_test

            # max_map = (max(max_map[0], result_print[0]), max(max_map[1], result_print[3]))
            # max_ndcg = (max(max_ndcg[0], result_print[1]), max(max_ndcg[1], result_print[4]))
            # max_prec = (max(max_prec[0], result_print[2]), max(max_prec[1], result_print[5]))

        self.model.save_model(f"D:/Code/graduation_design/ckpt/model.ckpt")

        # with open("./result.txt","a") as f:
        #     f.write("{},{},{},{},{},{},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f}\n".format(
        #         self.args.name,
        #         self.args.model,
        #         self.args.user_module,
        #         self.args.model_module,
        #         self.args.div_module,
        #         self.args.tradeoff,
        #         max_map[0],
        #         max_ndcg[0],
        #         max_prec[0],
        #         max_map[1],
        #         max_ndcg[1],
        #         max_prec[1]
        #         # result_print[0],
        #         # result_print[1],
        #         # result_print[2],
        #         # result_print[3],
        #         # result_print[4],
        #         # result_print[5]
        #     ))

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

        return result_map, result_ndcg, result_recall
        # return [
        #     np.mean(result_map[topk[0]]),
        #     np.mean(result_ndcg[topk[0]]),
        #     np.mean(result_recall[topk[0]]),
        #     np.mean(result_map[topk[1]]),
        #     np.mean(result_ndcg[topk[1]]),
        #     np.mean(result_recall[topk[1]])
        # ]
