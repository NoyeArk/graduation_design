import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn


class ContentExtractionModule(nn.Module):
    """
    商品内容提取模块
    """
    def __init__(self, pretrained_model="bert-base-chinese", max_length=512):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.llm = AutoModel.from_pretrained(pretrained_model)
        self.max_length = max_length

        # 平均池化层
        self.pooling = nn.AdaptiveAvgPool2d((1, self.llm.config.hidden_size))

    def forward(self, item_descriptions):
        """
        提取商品描述的内容表示
        
        Args:
            item_descriptions: List[str], 商品描述文本列表
            
        Returns:
            content_embeddings: torch.Tensor, 商品内容的嵌入表示
        """
        # 对输入文本进行编码
        encoded = self.tokenizer(
            item_descriptions,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # 获取语言模型的输出
        outputs = self.llm(**encoded)
        hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

        # 通过平均池化得到最终的内容表示
        content_embeddings = self.pooling(hidden_states.unsqueeze(1)).squeeze(1)

        return content_embeddings

    def extract_features(self, item_info):
        """
        从商品信息中提取特征
        
        Args:
            item_info: dict, 包含商品标题、品牌、类别等信息的字典

        Returns:
            str, 格式化的商品描述文本
        """
        desc = f"商品信息如下。商品标题是'{item_info['title']}'。"
        if 'category' in item_info:
            desc += f"该商品属于'{item_info['category']}'类别"
        if 'brand' in item_info:
            desc += f"品牌是'{item_info['brand']}'。"
        if 'year' in item_info:
            desc += f"发行年份是{item_info['year']}。"

        return desc

    @staticmethod
    def load_movielens_data(filepath="D:/Code/graduation_design/data/ml-1m/movies.dat", encoding='ISO-8859-1'):
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
                            'category': categories[0] if categories else None
                        }

                        if year:
                            movies_data[movie_id]['year'] = year
        except Exception as e:
            print(f"加载MovieLens数据失败: {e}")

        return movies_data


def main():
    # 使用示例
    model = ContentExtractionModule()

    # 加载MovieLens电影数据
    movies = model.load_movielens_data()
    print(f"成功加载 {len(movies)} 部电影")

    # 随机选择几部电影展示
    import random
    sample_ids = random.sample(list(movies.keys()), 3)

    embeddings_list = []
    for movie_id in sample_ids:
        movie_info = movies[movie_id]
        print(f"\n电影ID: {movie_id}")
        print(f"原始信息: {movie_info}")

        # 提取描述文本
        desc = model.extract_features(movie_info)
        print(f"格式化描述: {desc}")

        # 获取内容嵌入
        with torch.no_grad():
            embedding = model([desc])
            embeddings_list.append(embedding)

        print(f"内容嵌入维度: {embedding.shape}")


if __name__ == "__main__":
    main()
