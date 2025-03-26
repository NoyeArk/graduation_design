import torch
import torch.nn as nn
import math


class DIEN(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # GRU层 - 每个时间步一个GRU
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        
        # Attention层 - 每个时间步一个
        self.attention = AttentionLayer(hidden_dim, num_heads, dropout)
        
        # AUGRU层 - 每个时间步一个
        self.augru_cell = nn.GRUCell(hidden_dim * 2, hidden_dim)  # 输入维度翻倍因为要concat
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dim, 1)
        
    def forward(self, seq_emb, target_emb):
        """
        Args:
            seq_emb: 用户行为序列 [batch_size, seq_len, dim]
            target_emb: 目标物品 [batch_size, dim]
        """
        batch_size, seq_len, dim = seq_emb.size()

        # 1. 逐个时间步处理
        h_gru = torch.zeros(batch_size, self.hidden_dim, device=seq_emb.device)
        h_augru = torch.zeros(batch_size, self.hidden_dim, device=seq_emb.device)
        final_outputs = []
        
        for t in range(seq_len):
            # 获取当前时间步的物品embedding
            current_input = seq_emb[:, t, :]  # [batch_size, hidden_dim]
            
            # GRU处理
            gru_output = self.gru_cell(current_input, h_gru)  # [batch_size, hidden_dim]
            h_gru = gru_output  # 更新GRU隐状态
            
            # Attention处理
            # 将GRU输出扩展为序列形式以适应attention层
            gru_output_seq = gru_output.unsqueeze(1)  # [batch_size, 1, hidden_dim]
            attn_output = self.attention(
                query=target_emb,    # [batch_size, hidden_dim]
                key=gru_output_seq,     # [batch_size, 1, hidden_dim]
                value=gru_output_seq    # [batch_size, 1, hidden_dim]
            )  # [batch_size, hidden_dim]
            
            # 连接GRU输出和attention输出
            augru_input = torch.cat([gru_output, attn_output], dim=1)  # [batch_size, hidden_dim*2]
            
            # AUGRU处理
            augru_output = self.augru_cell(augru_input, h_augru)  # [batch_size, hidden_dim]
            h_augru = augru_output  # 更新AUGRU隐状态
            
            final_outputs.append(augru_output)
        
        # 2. 堆叠所有时间步的输出
        final_outputs = torch.stack(final_outputs, dim=1)  # [batch_size, seq_len, hidden_dim]
        final_hidden = final_outputs[:, -1, :]  # [batch_size, hidden_dim]

        return final_hidden  # [batch_size, hidden_dim]

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value):
        batch_size = query.size(0)
        seq_len = key.size(1)
        
        # 1. 线性投影
        q = self.q_proj(query)  # [batch_size, hidden_dim]
        k = self.k_proj(key)    # [batch_size, seq_len, hidden_dim]
        v = self.v_proj(value)  # [batch_size, seq_len, hidden_dim]
        
        # 2. 将query扩展为与key相同的序列长度
        q = q.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # 3. 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.hidden_dim)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 4. 加权求和
        attn_output = torch.matmul(attn_weights, v)  # [batch_size, 1, hidden_dim]
        
        return attn_output.squeeze(1)  # [batch_size, hidden_dim]

# 使用示例
def test_model():
    num_items = 1000
    embedding_dim = 64
    hidden_dim = 128
    batch_size = 32
    seq_len = 50
    
    model = DIEN(
        num_items=num_items,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim
    )

    seq = torch.randint(0, num_items, (batch_size, seq_len))
    target_items = torch.randint(0, num_items, (batch_size,))
    
    output = model(seq, target_items)
    print(f"输入序列形状: {seq.shape}")
    print(f"目标物品形状: {target_items.shape}")
    print(f"输出形状: {output.shape}")

if __name__ == "__main__":
    test_model()