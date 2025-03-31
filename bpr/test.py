import pandas as pd

class DataProcessor:
    @staticmethod
    def load_kuairec_data(filepath, encoding='utf-8'):
        """
        加载Kuairec数据集中的视频信息

        Args:
            filepath: str, Kuairec数据文件路径
            encoding: str, 文件编码格式

        Returns:
            dict, 视频ID到视频信息的映射字典
        """
        video_data = {}
        try:
            # 读取CSV文件
            df = pd.read_csv(filepath, encoding=encoding)

            for index, row in df.iterrows():
                video_id = row['video_id']
                manual_cover_text = row['manual_cover_text'] if pd.notna(row['manual_cover_text']) else ''
                caption = row['caption'] if pd.notna(row['caption']) else ''
                topic_tag = row['topic_tag'] if pd.notna(row['topic_tag']) else []
                first_level_category_name = row['first_level_category_name'] if pd.notna(row['first_level_category_name']) else None
                second_level_category_name = row['second_level_category_name'] if pd.notna(row['second_level_category_name']) else None
                third_level_category_name = row['third_level_category_name'] if pd.notna(row['third_level_category_name']) else None

                # 将视频信息存储到字典中
                video_data[video_id] = {
                    'manual_cover_text': manual_cover_text,
                    'caption': caption,
                    'topic_tag': topic_tag,
                    'first_level_category_name': first_level_category_name,
                    'second_level_category_name': second_level_category_name,
                    'third_level_category_name': third_level_category_name
                }

        except Exception as e:
            print(f"加载Kuairec数据失败: {e}")

        return video_data

# 示例用法
filepath = "D:/Code/graduation_design/data/kuairec/data/kuairec_caption_category.csv"
video_info = DataProcessor.load_kuairec_data(filepath)
print(video_info['10'])