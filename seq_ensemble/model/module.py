import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from seq_ensemble.utils import normalize


def moe(inputs):
    """
    实现 Mixture of Experts 模块
    
    Args:
        inputs (`tf.Tensor`): 输入张量, 形状为 [batch_size, n_experts, hidden_dim]

    Returns:
        outputs (`tf.Tensor`): 输出张量, 形状为 [batch_size, n_experts, hidden_dim]
    """
    n_experts = inputs.get_shape().as_list()[1]  # 专家数量(基模型数量)
    hidden_dim = inputs.get_shape().as_list()[2]  # 隐藏层维度

    # 1. 门控网络 - 为每个样本计算专家权重
    with tf.variable_scope("gate_network"):
        # 将输入展平
        flat_inputs = tf.reshape(inputs, [-1, n_experts * hidden_dim])
        # 门控网络层
        gate_weights = tf.layers.dense(
            flat_inputs,
            n_experts,
            activation=tf.nn.softmax,
            name="gate_weights"
        )  # [batch_size, n_experts]

    # 2. 专家网络 - 每个专家独立处理输入
    with tf.variable_scope("expert_network"):
        experts_outputs = []
        for i in range(n_experts):
            with tf.variable_scope(f"expert_{i}"):
                expert_input = inputs[:, i, :]  # [batch_size, hidden_dim]
                # 每个专家是一个两层前馈网络
                expert_hidden = tf.layers.dense(
                    expert_input,
                    hidden_dim,
                    activation=tf.nn.relu,
                    name="expert_hidden"
                )
                expert_output = tf.layers.dense(
                    expert_hidden,
                    hidden_dim,
                    activation=None,
                    name="expert_output"
                )  # [batch_size, hidden_dim]
                experts_outputs.append(expert_output)

        # 将所有专家输出堆叠
        experts_outputs = tf.stack(experts_outputs, axis=1)  # [batch_size, n_experts, hidden_dim]

    # 3. 组合专家输出
    # 扩展门控权重维度以便广播
    gate_weights = tf.expand_dims(gate_weights, axis=-1)  # [batch_size, n_experts, 1]

    # 加权组合专家输出
    weighted_outputs = experts_outputs * gate_weights  # [batch_size, n_experts, hidden_dim]

    # 4. 残差连接
    final_outputs = inputs + weighted_outputs

    # 5. Layer Normalization
    final_outputs = normalize(final_outputs)

    return final_outputs
