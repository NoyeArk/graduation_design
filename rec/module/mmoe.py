import torch
import torch.nn as nn
import torch.nn.functional as F


class MMOE(nn.Module):
    """
    Multi-gate Mixture-of-Experts 模型实现
    
    Args:
        input_dim (`int`): 输入特征维度
        num_experts (`int`): 专家数量
        expert_hidden_sizes (`list`): 专家网络隐藏层大小列表
        num_tasks (`int`): 任务数量
        task_hidden_sizes (`list`): 任务塔隐藏层大小列表
    """
    def __init__(self, input_dim, num_experts, expert_hidden_sizes, num_tasks, task_hidden_sizes):
        super(MMOE, self).__init__()
        
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.expert_hidden_sizes = expert_hidden_sizes
        self.num_tasks = num_tasks
        self.task_hidden_sizes = task_hidden_sizes
        
        # 创建专家网络
        self.experts = nn.ModuleList()
        for i in range(num_experts):
            expert = self._create_mlp(input_dim, expert_hidden_sizes)
            self.experts.append(expert)
        
        # 创建门控网络 - 每个任务一个门控网络
        self.gates = nn.ModuleList()
        for i in range(num_tasks):
            gate = nn.Linear(input_dim, num_experts)
            self.gates.append(gate)
        
        # 创建任务塔
        self.task_towers = nn.ModuleList()
        for i in range(num_tasks):
            tower = self._create_mlp(expert_hidden_sizes[-1], task_hidden_sizes, output_dim=1)
            self.task_towers.append(tower)
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(expert_hidden_sizes[-1])
        
    def _create_mlp(self, input_dim, hidden_sizes, output_dim=None):
        """创建多层感知机"""
        layers = []
        prev_dim = input_dim
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_size
        
        if output_dim is not None:
            layers.append(nn.Linear(prev_dim, output_dim))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入张量, 形状为 [batch_size, input_dim]
            
        Returns:
            list: 每个任务的输出列表, 每个元素形状为 [batch_size, 1]
        """
        # 获取批次大小
        batch_size = x.size(0)
        
        # 1. 专家网络前向传播
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        
        # 将专家输出堆叠 [batch_size, num_experts, expert_output_dim]
        expert_outputs = torch.stack(expert_outputs, dim=1)
        
        # 2. 为每个任务计算门控权重并组合专家输出
        task_outputs = []
        for task_id in range(self.num_tasks):
            # 计算门控权重
            gate_output = self.gates[task_id](x)  # [batch_size, num_experts]
            gate_output = F.softmax(gate_output, dim=1)  # [batch_size, num_experts]

            # 扩展门控权重维度以便广播
            gate_output = gate_output.unsqueeze(-1)  # [batch_size, num_experts, 1]

            # 加权组合专家输出
            weighted_expert_output = expert_outputs * gate_output  # [batch_size, num_experts, expert_output_dim]
            
            # 沿专家维度求和
            combined_expert_output = weighted_expert_output.sum(dim=1)  # [batch_size, expert_output_dim]
            
            # 应用层归一化
            normalized_output = self.layer_norm(combined_expert_output)
            
            # 3. 任务塔前向传播
            task_output = self.task_towers[task_id](normalized_output)  # [batch_size, 1]
            task_outputs.append(task_output)

        return task_outputs

# 使用示例
if __name__ == "__main__":
    # 参数设置
    input_dim = 128
    num_experts = 4
    expert_hidden_sizes = [64, 32]
    num_tasks = 2  # 例如：点击预测和转化预测
    task_hidden_sizes = [16, 8]

    # 创建模型
    model = MMOE(input_dim, num_experts, expert_hidden_sizes, num_tasks, task_hidden_sizes)
    
    # 模拟输入数据
    batch_size = 64
    x = torch.randn(batch_size, input_dim)
    
    # 前向传播
    outputs = model(x)
    
    # 打印输出形状
    for i, output in enumerate(outputs):
        print(f"Task {i} output shape: {output.shape}")
