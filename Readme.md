# 基于集成学习的个性化推荐算法设计

EnsRec（Ensemble Recommendation）是一种基于集成学习的个性化推荐算法。该项目旨在通过结合多种推荐模型的优势，提升推荐系统的性能和准确性。通过对多个基模型的结果进行集成，EnsRec能够更好地捕捉用户偏好和物品特性，从而生成更精准的推荐结果。

本项目的主要特点包括：

1. **多模型集成**：支持多种推荐算法的集成，包括ACF、ANAM、Caser、FDSA、HARNN、PFMC和SASRec。
2. **模块化设计**：项目结构清晰，便于扩展和维护。
3. **多数据集支持**：支持MovieLens-1M、KuaiRec和Toys_and_Games等多个数据集。
4. **实验验证**：通过消融实验和超参数搜索验证模型性能。

EnsRec的目标是为个性化推荐领域提供一种高效且易用的解决方案，适用于学术研究和实际应用场景。

![](image/ensrec.png)

实验结果：

![](image/result.png)

## 项目结构

```
.
├── basemodel        # 基模型部分实现
├── data             # 数据集
├── experiment       # 消融实验和超参搜索实验结果
│   ├── ablation
│   └── hyperparameter_learning
├── ipynb           # jupyter notebook文件
├── llm_emb         # 使用llm对不同数据集物品嵌入之后的结果
│   ├── KuaiRec
│   ├── MovieLens-1M
│   └── Toys_and_Games
├── Readme.md
├── requirements.txt
└── src
    ├── ckpt         # 模型权重
    ├── config.yaml  # 配置文件
    ├── data.py      # 数据处理文件
    ├── datasets     # 处理之后的数据集
    ├── infer.ipynb  # 推荐指标计算
    ├── main.py
    ├── model        # 模型实现
    └── module
```

## 安装与运行

### 准备环境

- 克隆项目到本地：

```bash
git clone https://github.com/NoyeArk/graduation_design.git
```

- 安装依赖：

```bash
pip install -r requirements.txt
```

### 准备数据集

本文使用的数据集为三个：MovieLens-1M、KuaiRec 和 Toys_and_Games，可以从[链接](https://pan.baidu.com/s/1ZgtYXfAwQELQcPSiYVkm_Q?pwd=d4a2)进行下载。也可以直接下载处理好的数据格式：[链接](https://pan.baidu.com/s/1lJTwDEFEw7JF6MXErxHaNA?pwd=ihs7)进行下载。

每个数据集由三个文件构成：

1. `interaction.csv`：包含用户-物品的交互
2. `user.csv`（未使用）：每个用户的描述信息
3. `item.csv`：每个物品的描述信息

### 计算基模型推荐结果

修改 `basemodel/config.yaml` 配置文件中的 `name` 字段为要运行的基模型 `[acf, anam, caser, fdsa, harnn, pfmc, sasrec]`。配置文件中的 `dataset` 下的 `path` 字段修改为要训练的数据集。

```bash
cd basemodel
python main.py
```

### 运行EnsRec模型

```bash
cd src
python main.py config.yaml
```
