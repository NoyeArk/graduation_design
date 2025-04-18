import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class TransformerBlock(nn.Module):
    """
    Transformer编码器块，不使用因果注意力
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
        # 自注意力层，不使用因果掩码
        residual = x
        x = self.layer_norm1(x)

        # 转置为注意力机制需要的形状 [seq_len, batch_size, hidden_size]
        x = x.transpose(0, 1)

        # 应用注意力
        if attention_mask is not None:
            # 转换注意力掩码形状
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0
        x, _ = self.attention(x, x, x, attn_mask=attention_mask)
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

        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=self.max_length,
                               padding="max_length", truncation=True)

        inputs = {k: v.to(self.llm.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.llm(**inputs)
            hidden_states = outputs.last_hidden_state

        content_embedding = torch.mean(hidden_states, dim=1)  # [batch_size, hidden_size]
        # content_embedding = hidden_states[:, -1, :]  # [batch_size, hidden_size]

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
    def __init__(self, hidden_factor, num_transformer_layers, num_attention_heads,
                 intermediate_size, max_seq_length, dropout_rate):
        super(PreferenceAlignmentModule, self).__init__()
        self.hidden_factor = hidden_factor
        self.max_seq_length = max_seq_length
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

    def forward(self, item_embedding):
        """
        前向传播
        
        Args:
            item_embedding (torch.Tensor): 内容嵌入序列，形状可以是
                1. [batch_size, n_base_model, seq_length, hidden_factor]
                2. [batch_size, seq_length, hidden_factor]

        Returns:
            refined_embeddings (torch.Tensor): 优化后的嵌入序列，形状与输入相同
        """
        if item_embedding.dim() == 4:
            # 如果输入是 [batch_size, n_base_model, seq_length, hidden_factor]
            batch_size, n_base_model, seq_length, hidden_dim = item_embedding.size()
            item_embedding = item_embedding.view(-1, seq_length, hidden_dim)
        else:
            # 输入是 [batch_size, seq_length, hidden_factor]
            batch_size, seq_length, hidden_dim = item_embedding.size()
            n_base_model = None

        # 应用内容适配器
        adapted_embeddings = self.content_adaptor(item_embedding)

        # 添加位置编码
        position_ids = torch.arange(seq_length, dtype=torch.long, device=adapted_embeddings.device)
        position_ids = position_ids.unsqueeze(0).expand(adapted_embeddings.size(0), -1)
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

        if n_base_model is not None:
            # 如果输入是 [batch_size, n_base_model, seq_length, hidden_factor]，恢复形状
            refined_embeddings = refined_embeddings.view(batch_size, n_base_model, seq_length, hidden_dim)
        
        return refined_embeddings


class ItemTower(nn.Module):
    def __init__(self, hidden_factor=64, pretrained_model_name="bert-base-uncased", max_length=128,
                 data_filepath="D:/Code/graduation_design/data/ml-1m/movies.dat",
                 cache_path="D:/Code/graduation_design/data/ml-1m/item_embeddings.npy",
                 device=None, num_transformer_layers=2, num_attention_heads=4,
                 intermediate_size=256, dropout_rate=0.1):
        super(ItemTower, self).__init__()
        self.device = device
        self.cache_path = cache_path
        self.hidden_factor = hidden_factor
        self.item_transform = nn.Sequential(
            nn.Linear(hidden_factor, hidden_factor),
            nn.ReLU(),
            nn.Linear(hidden_factor, hidden_factor)
        )

        self.preference_alignment = PreferenceAlignmentModule(
            hidden_factor=hidden_factor,
            num_transformer_layers=num_transformer_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_seq_length=max_length,
            dropout_rate=dropout_rate
        )

        self.layer_norm = nn.LayerNorm(hidden_factor)
        if data_filepath.split('/')[-2] == "ml-1m":
            self.item_data = self.load_movielens_data(data_filepath)
        elif data_filepath.split('/')[-2] == "kuairec":
            self.item_data = self.load_kuairec_data(data_filepath)
        elif data_filepath.split('/')[-2] == "Toys_and_Games":
            self.item_data = self.load_amazon_data(data_filepath)
        elif data_filepath.split('/')[-2] == "Office_Products":
            self.item_data = {}
            with open(data_filepath, 'r') as fp:
                for line in tqdm(fp):
                    data = line.strip().split('>>')
                    item_id = data[0]
                    if len(data) > 1:
                        self.item_data[item_id] = data[1]
                    else:
                        self.item_data[item_id] = ""
        else:
            raise ValueError(f"Unsupported dataset: {data_filepath}")

        self.item_to_idx = {str(item_id): idx for idx, item_id in enumerate(list(self.item_data.keys()))}
        self.item_to_idx['0'] = len(self.item_data)

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

    def _precompute_item_embeddings(self):
        self.cex = self.cex.to(self.device)
        self.item_transform.to(self.device)
        self.layer_norm.to(self.device)

        batch_size = 64
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

        with torch.no_grad():
            content_embeddings = self.cex([{"Unknown": "N/A"}], is_json=True)
            item_embeddings_batch = self.item_transform(content_embeddings)
            item_embeddings_batch = self.layer_norm(item_embeddings_batch)
            embeddings.append(item_embeddings_batch.cpu())

        self.item_embeddings = torch.cat(embeddings, dim=0).to(self.device)
        np.save(self.cache_path, self.item_embeddings.cpu().numpy())
        print(f">>>> 物品内容嵌入已保存到: {self.cache_path}, 形状: {self.item_embeddings.shape}")

    def _precompute_movie_embeddings(self):
        print(">>>> 预计算物品内容嵌入...")
        if hasattr(self.cex, 'is_meta') and self.cex.is_meta:
            self.cex = self.cex.to_empty(device=self.device)
        else:
            self.cex = self.cex.to(self.device)
        self.item_transform.to(self.device)
        self.layer_norm.to(self.device)

        batch_size = 32
        embeddings = []

        items = list(self.item_data.keys())
        for i in tqdm(range(0, len(items), batch_size), desc="预计算物品嵌入"):
            batch_ids = items[i:i+batch_size]
            batch_descriptions = []

            for item_id in batch_ids:
                item_info = self.item_data[item_id]
                batch_descriptions.append(item_info)

            with torch.no_grad():
                content_embeddings = self.cex(batch_descriptions)  # 使用forward方法处理批次
                content_embeddings = content_embeddings.to(self.device)
                item_embeddings_batch = self.item_transform(content_embeddings)
                item_embeddings_batch = self.layer_norm(item_embeddings_batch)
                embeddings.append(item_embeddings_batch.cpu())

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

        self.item_embeddings = torch.cat(embeddings, dim=0).to(self.device)
        np.save(self.cache_path, self.item_embeddings.cpu().numpy())
        print(f">>>> 物品内容嵌入已保存到: {self.cache_path}, 形状: {self.item_embeddings.shape}")

    @staticmethod
    def load_amazon_data(filepath, encoding='utf-8'):
        item_data = {}
        with open(filepath, 'r', encoding=encoding) as fp:
            for line in tqdm(fp):
                data = json.loads(line.strip())
                item_id = data['parent_asin']
                item_data[item_id] = {
                    "main_category": data['main_category'],
                    "title": data['title'],
                    "average_rating": data['average_rating'],
                    "rating_number": data['rating_number'],
                    "features": data['features'],
                    "description": data['description'],
                    "price": data['price'],
                    "images": data['images'],
                    "store": data['store'],
                    "categories": data['categories'],
                    "details": data['details']
                }
        return item_data

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

        return movies_data
    
    @staticmethod
    def load_kuairec_data(filepath, encoding='utf-8'):
        df = pd.read_csv(filepath, encoding=encoding)
        video_data = {}

        for index, row in df.iterrows():
            video_id = row['video_id']
            manual_cover_text = row['manual_cover_text'] if pd.notna(row['manual_cover_text']) else ''
            caption = row['caption'] if pd.notna(row['caption']) else ''
            topic_tag = row['topic_tag'] if pd.notna(row['topic_tag']) else []
            first_level_category_name = row['first_level_category_name'] if pd.notna(row['first_level_category_name']) else None
            second_level_category_name = row['second_level_category_name'] if pd.notna(row['second_level_category_name']) else None
            third_level_category_name = row['third_level_category_name'] if pd.notna(row['third_level_category_name']) else None

            video_data[video_id] = {
                'manual_cover_text': manual_cover_text,
                'caption': caption,
                'topic_tag': topic_tag,
                'first_level_category_name': first_level_category_name,
                'second_level_category_name': second_level_category_name,
                'third_level_category_name': third_level_category_name
            }

        return video_data

    def forward(self, item_ids, type):
        """
        前向传播

        Args:
            item_ids (torch.Tensor): 物品ID, [1, 3952]
            type (str): 类型, 'base_model' / 'user_seq' / 'single_item'

        Returns:
            item_embedding (torch.Tensor): 物品嵌入，形状为 [batch_size, seq, hidden_factor]，其中 batch_size 为批次大小，seq 为序列长度，hidden_factor 为隐藏层维度
        """
        if type == 'base_model':  # [bc, n_base_model, seq_len]
            batch_size, n_base_model, seq_len = item_ids.shape
            item_ids_np = item_ids.cpu().numpy()
            item_ids_flat = item_ids_np.reshape(-1)
            # item_ids_mapped = np.array([self.item_to_idx.get(str(self.id_to_item[id]), len(self.item_data)) for id in item_ids_flat])
            item_ids_mapped = np.array([self.item_to_idx[str(id)] for id in item_ids_flat])
            item_indices = torch.tensor(item_ids_mapped, device=item_ids.device)
            item_embeddings = self.item_embeddings[item_indices]
            item_embeddings = item_embeddings.reshape(batch_size, n_base_model, seq_len, -1)
            item_embeddings = self.preference_alignment(item_embeddings)

        elif type == 'user_seq':  # [bc, seq_len]
            batch_size, seq_len = item_ids.shape
            item_ids_np = item_ids.cpu().numpy()
            item_ids_flat = item_ids_np.reshape(-1)
            # item_ids_mapped = np.array([self.item_to_idx.get(str(self.id_to_item[id]), len(self.item_data)) for id in item_ids_flat])
            item_ids_mapped = np.array([self.item_to_idx[str(id)] for id in item_ids_flat])
            item_indices = torch.tensor(item_ids_mapped, device=item_ids.device)
            item_embeddings = self.item_embeddings[item_indices]
            item_embeddings = item_embeddings.reshape(batch_size, seq_len, -1)
            item_embeddings = self.preference_alignment(item_embeddings)

        elif type == 'single_item':  # [bc]
            item_ids_np = item_ids.cpu().numpy()
            # item_ids_mapped = np.array([self.item_to_idx.get(str(self.id_to_item[id]), len(self.item_data)) for id in item_ids_np])
            item_ids_mapped = np.array([self.item_to_idx[str(id)] for id in item_ids_np])
            item_indices = torch.tensor(item_ids_mapped, device=item_ids.device)
            item_embeddings = self.item_embeddings[item_indices]

        item_embeddings = self.item_transform(item_embeddings)
        item_embeddings = self.layer_norm(item_embeddings)

        return item_embeddings


if __name__ == "__main__":
    item_tower = ItemTower(hidden_factor=64, pretrained_model_name="bert-base-uncased", max_length=128,
                           data_filepath="D:/Code/graduation_design/data/Toys_and_Games/item.csv",
                           cache_path="D:/Code/graduation_design/llm_emb/Toys_and_Games/bert_emb64.npy",
                           device="cuda", num_transformer_layers=2, num_attention_heads=4,
                           intermediate_size=256, dropout_rate=0.1)
