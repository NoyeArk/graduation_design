import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class TransformerBlock(nn.Module):
    """
    Transformer编码器块，使用因果注意力
    """
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_attention_heads, dropout=dropout_rate)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size)
        )
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attention_mask=None):
        # 自注意力层，使用因果掩码
        residual = x
        x = self.layer_norm1(x)

        # 转置为注意力机制需要的形状 [seq_len, batch_size, hidden_size]
        x = x.transpose(0, 1)
        
        # 创建因果掩码
        seq_len = x.size(0)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), diagonal=1)
        
        # 应用注意力
        if attention_mask is not None:
            # 组合因果掩码和注意力掩码
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0
            causal_mask = causal_mask + attention_mask
            
        x, _ = self.attention(x, x, x, attn_mask=causal_mask)
        x = x.transpose(0, 1)  # 转回 [batch_size, seq_len, hidden_size]
        x = self.dropout(x)
        x = residual + x
        
        # 前馈网络
        residual = x
        x = self.layer_norm2(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = residual + x

        return x


class ContentExtractionModule(nn.Module):
    """
    内容提取模块 (CEX)
    使用预训练的LLM和平均池化层处理项目描述，生成内容嵌入
    """
    def __init__(self, hidden_factor=64, pretrained_model_name="roberta-large", max_length=128):
        super(ContentExtractionModule, self).__init__()
        self.hidden_factor = hidden_factor
        self.pretrained_model_name = pretrained_model_name
        self.max_length = max_length
        self.llm = AutoModel.from_pretrained(
            pretrained_model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        # self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        # self.llm = BertModel.from_pretrained(pretrained_model_name)

        # 如果LLM的隐藏维度与目标维度不同，添加一个线性层
        self.projection = None
        if self.llm.config.hidden_size != hidden_factor:
            self.projection = nn.Linear(self.llm.config.hidden_size, hidden_factor)

    def process_item_description(self, item_description, is_json=False):
        """
        处理单个项目描述，生成内容嵌入

        Args:
            item_description (dict): 包含项目信息的字典

        Returns:
            content_embedding (torch.Tensor): 内容嵌入向量
        """
        if not is_json:
            prompt = f"""
                The item information is given as follows. Item title is "{item_description['title']}".
                This item belongs to "{item_description['category']}" and brand is "{item_description['brand']}".
                The price is "{item_description['price']}". The words of item are "{item_description['keywords']}".
                This item supports "{item_description['features']}".
            """
        else:
            prompt = f"""
                The item information is given as follows. {item_description}".
            """

        # 对文本进行编码
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=self.max_length,
                               padding="max_length", truncation=True)

        inputs = {k: v.to(self.llm.device) for k, v in inputs.items()}

        # 获取LLM的最终隐藏状态
        with torch.no_grad():
            outputs = self.llm(**inputs)
            hidden_states = outputs.last_hidden_state

        # 平均池化
        content_embedding = torch.mean(hidden_states, dim=1)  # [batch_size, hidden_size]

        # 如果需要，进行维度投影
        if self.projection is not None:
            content_embedding = content_embedding.float()  # 将 BFloat16 转换为 Float
            content_embedding = self.projection(content_embedding)

        return content_embedding

    def forward(self, item_descriptions, is_json=False):
        """
        处理一批项目描述
        
        Args:
            item_descriptions (list): 项目描述列表
            is_json (bool): 是否为json格式

        Returns:
            content_embeddings (torch.Tensor): 内容嵌入张量 [batch_size, hidden_factor]
        """
        content_embeddings = []
        for item_desc in item_descriptions:
            embedding = self.process_item_description(item_desc, is_json)
            content_embeddings.append(embedding)

        return torch.cat(content_embeddings, dim=0)


class PreferenceAlignmentModule(nn.Module):
    """
    偏好对齐模块 (PAL)
    捕捉用户偏好，并根据内容嵌入序列输出用户嵌入
    """
    def __init__(self, hidden_factor=64, num_transformer_layers=12, num_attention_heads=12,
                 intermediate_size=3072, max_seq_length=50, dropout_rate=0.1):
        super(PreferenceAlignmentModule, self).__init__()
        self.hidden_factor = hidden_factor
        self.max_seq_length = max_seq_length

        # 内容适配器 - 维度转换
        self.content_adaptor = nn.Linear(hidden_factor, hidden_factor)

        # 位置编码
        self.position_embeddings = nn.Embedding(max_seq_length, hidden_factor)

        # Transformer层
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(
                hidden_factor,
                num_attention_heads,
                intermediate_size,
                dropout_rate
            ) for _ in range(num_transformer_layers)
        ])

        # 在线投影层
        self.online_projection = nn.Linear(hidden_factor, hidden_factor)

        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_factor)

    def forward(self, content_embedding_sequence):
        """
        前向传播
        
        Args:
            content_embedding_sequence (torch.Tensor): 内容嵌入序列 [batch_size, seq_length, hidden_factor]

        Returns:
            refined_embeddings (torch.Tensor): 优化后的嵌入序列 [batch_size, seq_length, hidden_factor]
        """
        batch_size, seq_length, _ = content_embedding_sequence.size()

        # 应用内容适配器
        adapted_embeddings = self.content_adaptor(content_embedding_sequence)

        # 添加位置编码
        position_ids = torch.arange(seq_length, dtype=torch.long, device=adapted_embeddings.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embeddings(position_ids)
        
        embeddings = adapted_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        # 应用Transformer层，不使用注意力掩码
        hidden_states = embeddings
        for layer in self.transformer_layers:
            hidden_states = layer(hidden_states)

        # 保持序列维度，应用在线投影层到每个时间步
        refined_embeddings = self.online_projection(hidden_states)
        
        return refined_embeddings


class ItemTower(nn.Module):
    """
    物品塔，处理物品特征并生成物品嵌入
    """
    def __init__(self, hidden_factor=64, pretrained_model_name="bert-base-uncased", max_length=128,
                 data_filepath="D:/Code/graduation_design/data/ml-1m/movies.dat",
                 cache_path="D:/Code/graduation_design/data/ml-1m/item_embeddings.npy",
                 device=None, num_transformer_layers=2, num_attention_heads=4,
                 intermediate_size=256, dropout_rate=0.1):
        super(ItemTower, self).__init__()
        self.device = device
        self.cache_path = cache_path
        self.hidden_factor = hidden_factor

        # 物品特征转换层 - 确保输入维度与LLM输出维度匹配
        self.item_transform = nn.Sequential(
            nn.Linear(hidden_factor, hidden_factor),
            nn.ReLU(),
            nn.Linear(hidden_factor, hidden_factor)
        )

        # 添加PreferenceAlignmentModule用于进一步处理物品嵌入
        self.preference_alignment = PreferenceAlignmentModule(
            hidden_factor=hidden_factor,
            num_transformer_layers=num_transformer_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_seq_length=max_length,  # 默认最大序列长度
            dropout_rate=dropout_rate
        )

        self.layer_norm = nn.LayerNorm(hidden_factor)
        self.item_data = self.load_movielens_data(data_filepath)
        if data_filepath == "D:/Code/graduation_design/data/magazine/item.dat":
            self.cex = ContentExtractionModule(
                hidden_factor=hidden_factor,
                pretrained_model_name=pretrained_model_name,
                max_length=max_length
            )
            self.item_data = {}
            with open(data_filepath, 'r') as fp:
                for line in tqdm(fp):
                    item_id = line.strip().split('::')[0]
                    self.item_data[item_id] = line.strip().split('::')[1]
        self.item_to_idx = {item_id: idx for idx, item_id in enumerate(list(self.item_data.keys()))}
        self.item_to_idx['0'] = len(self.item_data)

        # 如果缓存路径不存在，则预计算物品嵌入
        if not os.path.exists(self.cache_path):
            self.cex = ContentExtractionModule(
                hidden_factor=hidden_factor,
                pretrained_model_name=pretrained_model_name,
                max_length=max_length
            )
            self._precompute_item_embeddings()
        else:
            print(">>>> 加载预计算的物品嵌入...")
            self.item_embeddings = torch.from_numpy(np.load(self.cache_path)).float().to(self.device)

    def _precompute_magazine_item_embeddings(self):
        self.cex = self.cex.to(self.device)
        self.item_transform.to(self.device)
        self.layer_norm.to(self.device)

        batch_size = 32
        embeddings = []

        items = list(self.item_data.keys())
        for i in tqdm(range(0, len(items), batch_size), desc=">>>> 预计算物品嵌入"):
            batch_ids = items[i:i+batch_size]
            batch_descriptions = []

            for item_id in batch_ids:
                item_info = self.item_data[item_id]  # json 格式
                batch_descriptions.append(item_info)

            with torch.no_grad():
                content_embeddings = self.cex(batch_descriptions, is_json=True)
                content_embeddings = content_embeddings.to(self.device)
                item_embeddings_batch = self.item_transform(content_embeddings)
                item_embeddings_batch = self.layer_norm(item_embeddings_batch)
                embeddings.append(item_embeddings_batch.cpu())

        # 如果物品id超出范围，使用一个默认的模型嵌入
        with torch.no_grad():
            content_embeddings = self.cex([{"Unknown": "N/A"}], is_json=True)
            item_embeddings_batch = self.item_transform(content_embeddings)
            item_embeddings_batch = self.layer_norm(item_embeddings_batch)
            embeddings.append(item_embeddings_batch.cpu())

        self.item_embeddings = torch.cat(embeddings, dim=0).to(self.device)
        np.save(self.cache_path, self.item_embeddings.cpu().numpy())
        print(f">>>> 物品内容嵌入已保存到: {self.cache_path}, 形状: {self.item_embeddings.shape}")

    def _precompute_item_embeddings(self):
        """
        预计算所有电影的内容嵌入
        """
        print(">>>> 预计算物品内容嵌入...")
        if hasattr(self.cex, 'is_meta') and self.cex.is_meta:
            self.cex = self.cex.to_empty(device=self.device)
        else:
            self.cex = self.cex.to(self.device)
        self.item_transform.to(self.device)
        self.layer_norm.to(self.device)

        batch_size = 32
        embeddings = []

        # 获取所有电影ID
        items = list(self.item_data.keys())
        for i in tqdm(range(0, len(items), batch_size), desc="预计算物品嵌入"):
            batch_ids = items[i:i+batch_size]
            batch_descriptions = []

            for item_id in batch_ids:
                item_info = self.item_data.get(item_id, {'title': f'未知电影{item_id}', 'category': '未知类别'})
                # 确保movie_info包含所有必要的键
                if 'brand' not in item_info:
                    item_info['brand'] = 'Unknown'
                if 'price' not in item_info:
                    item_info['price'] = 'N/A'
                if 'keywords' not in item_info:
                    item_info['keywords'] = item_info.get('category', 'Unknown')
                if 'features' not in item_info:
                    item_info['features'] = 'standard viewing'

                batch_descriptions.append(item_info)

            with torch.no_grad():
                # 获取内容嵌入
                content_embeddings = self.cex(batch_descriptions)  # 使用forward方法处理批次
                # 确保content_embeddings在正确的设备上
                content_embeddings = content_embeddings.to(self.device)

                # 应用物品特征转换
                item_embeddings_batch = self.item_transform(content_embeddings)
                item_embeddings_batch = self.layer_norm(item_embeddings_batch)

                # 添加到结果列表
                embeddings.append(item_embeddings_batch.cpu())

        # 如果物品id超出范围，使用一个默认的模型嵌入
        movie_info = {
            'title': 'Unknown', 
            'category': 'Unknown',
            'brand': 'Unknown',
            'price': 'N/A',
            'keywords': 'Unknown',
            'features': 'standard viewing'
        }
        with torch.no_grad():
            content_embeddings = self.cex([movie_info])
            item_embeddings_batch = self.item_transform(content_embeddings)
            item_embeddings_batch = self.layer_norm(item_embeddings_batch)
            embeddings.append(item_embeddings_batch.cpu())

        # 合并所有批次的嵌入
        self.item_embeddings = torch.cat(embeddings, dim=0).to(self.device)
        np.save(self.cache_path, self.item_embeddings.cpu().numpy())
        print(f">>>> 物品内容嵌入已保存到: {self.cache_path}, 形状: {self.item_embeddings.shape}")

    @staticmethod
    def load_movielens_data(filepath, encoding='ISO-8859-1'):
        """
        加载MovieLens数据集中的电影信息

        Args:
            filepath: str, MovieLens数据文件路径
            encoding: str, 文件编码格式

        Returns:
            dict, 电影ID到电影信息的映射字典
        """
        movies_data = {}
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                for line in f:
                    parts = line.strip().split('::')
                    if len(parts) == 3:
                        movie_id = parts[0]
                        title_with_year = parts[1]
                        categories = parts[2].split('|')

                        # 从标题中提取年份
                        year = None
                        title = title_with_year
                        if '(' in title_with_year and ')' in title_with_year:
                            year_start = title_with_year.rfind('(')
                            year_end = title_with_year.rfind(')')
                            if year_start < year_end:
                                year_str = title_with_year[year_start+1:year_end]
                                if year_str.isdigit():
                                    year = year_str
                                    title = title_with_year[:year_start].strip()

                        movies_data[movie_id] = {
                            'title': title,
                            'category': categories[0] if categories else None,
                            'keywords': ', '.join(categories)  # 使用类别作为关键词
                        }

                        if year:
                            movies_data[movie_id]['year'] = year
        except Exception as e:
            print(f"加载MovieLens数据失败: {e}")

        return movies_data

    def forward(self, item_ids, type):
        """
        前向传播

        Args:
            item_id (torch.Tensor): 物品ID, [1, 3952]
            type (str): 类型, 'base_model' / 'user_seq' / 'single_item'

        Returns:
            item_embedding (torch.Tensor): 物品嵌入，形状为 [batch_size, seq, hidden_factor]，其中 batch_size 为批次大小，seq 为序列长度，hidden_factor 为隐藏层维度
        """
        if type == 'base_model':
            batch_size, n_base_model, seq_len = item_ids.shape
            item_ids_np = item_ids.cpu().numpy()
            item_ids_flat = item_ids_np.reshape(-1)
            item_ids_mapped = np.array([self.item_to_idx[str(id)] for id in item_ids_flat])
            item_indices = torch.tensor(item_ids_mapped, device=item_ids.device)
            item_embeddings = self.item_embeddings[item_indices]
            item_embeddings = item_embeddings.reshape(batch_size, n_base_model, seq_len, -1)

        elif type == 'user_seq':
            batch_size, seq_len = item_ids.shape
            item_ids_np = item_ids.cpu().numpy()
            item_ids_flat = item_ids_np.reshape(-1)
            item_ids_mapped = np.array([self.item_to_idx[str(id)] for id in item_ids_flat])
            item_indices = torch.tensor(item_ids_mapped, device=item_ids.device)
            item_embeddings = self.item_embeddings[item_indices]
            item_embeddings = item_embeddings.reshape(batch_size, seq_len, -1)

        elif type == 'single_item':
            item_ids_np = item_ids.cpu().numpy()
            item_ids_mapped = np.array([self.item_to_idx[str(id)] for id in item_ids_np])
            item_indices = torch.tensor(item_ids_mapped, device=item_ids.device)
            item_embeddings = self.item_embeddings[item_indices]

        item_embeddings = self.item_transform(item_embeddings)
        item_embeddings = self.layer_norm(item_embeddings)

        return item_embeddings


if __name__ == "__main__":
    item_tower = ItemTower(hidden_factor=64, pretrained_model_name="bert-base-uncased", max_length=128,
                           data_filepath="D:/Code/graduation_design/data/magazine/item.dat",
                           cache_path="D:/Code/graduation_design/llm_emb/magazine/bert_emb64.npy",
                           device="cuda", num_transformer_layers=2, num_attention_heads=4,
                           intermediate_size=256, dropout_rate=0.1)
    item_tower._precompute_magazine_item_embeddings()
