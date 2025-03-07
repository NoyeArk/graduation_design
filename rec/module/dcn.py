import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossNetwork(nn.Module):
    """
    DCN中的Cross Network部分
    
    Args:
        input_dim (int): 输入特征维度
        num_layers (int): 交叉层数量
        l2_reg (float): L2正则化系数
    """
    def __init__(self, input_dim, num_layers=2, l2_reg=0.001):
        super(CrossNetwork, self).__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        
        # 创建每层的权重和偏置
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim, 1) * 0.01)
            for _ in range(num_layers)
        ])
        
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim, 1))
            for _ in range(num_layers)
        ])
        
        # 应用L2正则化
        self.l2_reg = l2_reg
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入张量, 形状为 [batch_size, input_dim]
            
        Returns:
            torch.Tensor: 输出张量, 形状为 [batch_size, input_dim]
        """
        # 调整输入形状为 [batch_size, input_dim, 1]
        x_0 = x.unsqueeze(2)  # [batch_size, input_dim, 1]
        x_l = x_0  # [batch_size, input_dim, 1]
        
        for i in range(self.num_layers):
            # xl_w = x_l^T * w_l, 形状为 [batch_size, 1, 1]
            xl_w = torch.matmul(x_l.transpose(1, 2), self.weights[i])  # [batch_size, 1, 1]
            
            # x_0 * xl_w, 形状为 [batch_size, input_dim, 1]
            dot_ = torch.matmul(x_0, xl_w)  # [batch_size, input_dim, 1]
            
            # x_{l+1} = x_0 * (x_l^T * w_l) + b_l + x_l
            x_l = dot_ + self.biases[i] + x_l  # [batch_size, input_dim, 1]
        
        # 调整输出形状为 [batch_size, input_dim]
        return x_l.squeeze(2)  # [batch_size, input_dim]
    
    def get_regularization_loss(self):
        """计算L2正则化损失"""
        reg_loss = 0.0
        for w in self.weights:
            reg_loss += torch.sum(torch.square(w))
        for b in self.biases:
            reg_loss += torch.sum(torch.square(b))
        return self.l2_reg * reg_loss


class DeepNetwork(nn.Module):
    """
    DCN中的Deep Network部分
    
    Args:
        input_dim (int): 输入特征维度
        hidden_units (list): 隐藏层单元数列表
        dropout_rate (float): Dropout比率
        use_bn (bool): 是否使用Batch Normalization
    """
    def __init__(self, input_dim, hidden_units, dropout_rate=0.0, use_bn=False):
        super(DeepNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        
        # 构建MLP层
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_units:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
                
            layers.append(nn.ReLU())
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
                
            prev_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入张量, 形状为 [batch_size, input_dim]
            
        Returns:
            torch.Tensor: 输出张量, 形状为 [batch_size, hidden_units[-1]]
        """
        return self.mlp(x)


class DCN(nn.Module):
    """
    Deep & Cross Network模型
    
    Args:
        input_dim (int): 输入特征维度
        cross_num_layers (int): 交叉网络层数
        deep_hidden_units (list): 深度网络隐藏层单元数列表
        dropout_rate (float): Dropout比率
        use_bn (bool): 是否使用Batch Normalization
        output_dim (int): 输出维度，默认为1（二分类）
    """
    def __init__(self, input_dim, cross_num_layers=2, deep_hidden_units=[128, 64], 
                 dropout_rate=0.0, use_bn=False, output_dim=1):
        super(DCN, self).__init__()
        self.input_dim = input_dim
        self.cross_num_layers = cross_num_layers
        self.deep_hidden_units = deep_hidden_units
        
        # 交叉网络
        self.cross_net = CrossNetwork(input_dim, cross_num_layers)
        
        # 深度网络
        self.deep_net = DeepNetwork(input_dim, deep_hidden_units, dropout_rate, use_bn)
        
        # 组合层
        self.combined_dim = input_dim + deep_hidden_units[-1]  # 交叉网络输出 + 深度网络输出
        self.output_layer = nn.Linear(self.combined_dim, output_dim)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入张量, 形状为 [batch_size, input_dim]
            
        Returns:
            torch.Tensor: 输出张量, 形状为 [batch_size, output_dim]
        """
        # 交叉网络前向传播
        cross_output = self.cross_net(x)  # [batch_size, input_dim]
        
        # 深度网络前向传播
        deep_output = self.deep_net(x)  # [batch_size, deep_hidden_units[-1]]
        
        # 组合交叉网络和深度网络的输出
        combined = torch.cat([cross_output, deep_output], dim=1)  # [batch_size, combined_dim]
        
        # 输出层
        output = self.output_layer(combined)  # [batch_size, output_dim]
        
        return output
    
    def get_regularization_loss(self):
        """获取模型的正则化损失"""
        return self.cross_net.get_regularization_loss()


# 使用示例
if __name__ == "__main__":
    # 参数设置
    input_dim = 64
    cross_num_layers = 3
    deep_hidden_units = [128, 64, 32]
    dropout_rate = 0.2
    use_bn = True
    output_dim = 1  # 二分类问题
    
    # 创建模型
    model = DCN(input_dim, cross_num_layers, deep_hidden_units, dropout_rate, use_bn, output_dim)
    
    # 模拟输入数据
    batch_size = 32
    x = torch.randn(batch_size, input_dim)
    
    # 前向传播
    output = model(x)
    
    # 打印输出形状
    print(f"Output shape: {output.shape}")  # 应该是 [32, 1]
    
    # 计算正则化损失
    reg_loss = model.get_regularization_loss()
    print(f"Regularization loss: {reg_loss.item()}")