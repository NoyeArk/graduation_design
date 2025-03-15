# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

def positional_encoding(dim, sentence_length, dtype=tf.float32):
    encoded_vec = np.array([pos/np.power(10000, 2*i/dim) for pos in range(sentence_length) for i in range(dim)])
    encoded_vec[::2] = np.sin(encoded_vec[::2])
    encoded_vec[1::2] = np.cos(encoded_vec[1::2])

    return tf.convert_to_tensor(encoded_vec.reshape([sentence_length, dim]), dtype=dtype)

def normalize(inputs, epsilon=1e-8, scope="ln", reuse=None):
    '''应用层归一化。
    
    Args:
      inputs: 一个2维或更高维的张量,第一维为batch_size。
      epsilon: 一个很小的浮点数,用于防止除零错误。
      scope: variable_scope的可选作用域。
      reuse: 是否重用具有相同名称的前一层的权重。
      
    Returns:
      一个与inputs具有相同形状和数据类型的张量。
    '''
    inputs_shape = inputs.shape
    params_shape = inputs_shape[-1:]

    mean, variance = tf.nn.moments(inputs, [-1], keepdims=True)
    beta = tf.Variable(tf.zeros(params_shape))
    gamma = tf.Variable(tf.ones(params_shape))
    normalized = (inputs - mean) / ((variance + epsilon) ** 0.5)
    outputs = gamma * normalized + beta

    return outputs

def embedding(inputs,
              vocab_size,
              num_units,
              zero_pad=True,
              scale=True,
              l2_reg=0.0,
              scope="embedding",
              with_t=False,
              reuse=None):
    '''Embeds a given tensor.

    Args:
      inputs: 一个类型为`int32`或`int64`的`Tensor`，包含需要在`lookup table`中查找的id。
      vocab_size: 一个整数。词汇表大小。
      num_units: 一个整数。嵌入隐藏单元的数量。
      zero_pad: 一个布尔值。如果为True, 第一行(id 0)的所有值应该为常数0。
      scale: 一个布尔值。如果为True, 输出会乘以sqrt(num_units)。
      scope: `variable_scope`的可选作用域。
      reuse: 布尔值，是否重用具有相同名称的前一层的权重。

    Returns:
      A `Tensor`, 其秩比输入多1。最后一维应为`num_units`。
        
    For example,
    
    ```
    import tensorflow as tf
    
    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[ 0.          0.        ]
      [ 0.09754146  0.67385566]
      [ 0.37864095 -0.35689294]]

     [[-1.01329422 -1.09939694]
      [ 0.7521342   0.38203377]
      [-0.04973143 -0.06210355]]]
    ```
    
    ```
    import tensorflow as tf
    
    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[-0.19172323 -0.39159766]
      [-0.43212751 -0.66207761]
      [ 1.03452027 -0.26704335]]

     [[-0.11634696 -0.35983452]
      [ 0.50208133  0.53509563]
      [ 1.22204471 -0.96587461]]]    
    ```    
    '''
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        lookup_table = tf.compat.v1.get_variable(
            'lookup_table',
            dtype=tf.float32,
            shape=[vocab_size, num_units],
            #initializer=tf.keras.initializers.GlorotUniform(),
            regularizer=tf.keras.regularizers.l2(l2_reg)
        )
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)

        if scale:
            outputs = outputs * (num_units ** 0.5)
    if with_t:
        return outputs,lookup_table
    else:
        return outputs

def multihead_attention(queries, keys, num_units=None, num_heads=8, dropout_rate=0,
                       is_training=True, causality=False, scope="multihead_attention", with_qk=False):
    '''应用多头注意力机制。
    
    Args:
      queries: 形状为[N, T_q, C_q]的3维张量。
      keys: 形状为[N, T_k, C_k]的3维张量。
      num_units: 注意力大小的标量。
      dropout_rate: dropout比率。
      is_training: 是否处于训练模式。
      causality: 如果为True,会屏蔽未来的信息。
      num_heads: 注意力头的数量。
      scope: variable_scope的可选作用域。
      with_qk: 是否返回Q和K。
      
    Returns:
      形状为(N, T_q, C)的3维张量。
    '''
    if num_units is None:
        num_units = queries.shape[-1]
    
    # 线性投影
    Q = tf.keras.layers.Dense(num_units)(queries)  # (N, T_q, C)
    K = tf.keras.layers.Dense(num_units)(keys)  # (N, T_k, C)
    V = tf.keras.layers.Dense(num_units)(keys)  # (N, T_k, C)

    # 分割和拼接
    Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
    K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
    V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

    # 计算注意力分数
    outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

    # 缩放
    outputs = outputs / (K_.shape[-1] ** 0.5)

    # Key Masking
    key_masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1))  # (N, T_k)
    key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
    key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

    paddings = tf.ones_like(outputs) * (-2**32+1)
    outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

    # 因果关系掩码
    if causality:
        diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
        tril = tf.linalg.band_part(diag_vals, -1, 0)  # (T_q, T_k)
        masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)
        paddings = tf.ones_like(masks) * (-2**32+1)
        outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

    # 激活
    outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

    # Query Masking
    query_masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))  # (N, T_q)
    query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
    query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
    outputs *= query_masks  # (h*N, T_q, T_k)

    # Dropout
    outputs = tf.keras.layers.Dropout(dropout_rate)(outputs, training=is_training)

    # 加权求和
    outputs = tf.matmul(outputs, V_)  # (h*N, T_q, C/h)

    # 恢复形状
    outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

    # 残差连接
    outputs += queries

    if with_qk:
        return Q, K
    return outputs


def feedforward(inputs, num_units=[2048, 512], scope="feedforward", dropout_rate=0.2, is_training=True):
    '''逐点前馈网络。
    
    Args:
      inputs: 形状为[N, T, C]的3维张量。
      num_units: 两个整数的列表。
      scope: variable_scope的可选作用域。
      dropout_rate: dropout比率。
      is_training: 是否处于训练模式。
      
    Returns:
      一个与inputs具有相同形状和数据类型的3维张量。
    '''
    # 第一层卷积
    outputs = tf.keras.layers.Conv1D(
        filters=num_units[0],
        kernel_size=1,
        activation=tf.nn.relu,
        use_bias=True
    )(inputs)
    outputs = tf.keras.layers.Dropout(dropout_rate)(outputs, training=is_training)

    # 第二层卷积
    outputs = tf.keras.layers.Conv1D(
        filters=num_units[1],
        kernel_size=1,
        activation=None,
        use_bias=True
    )(outputs)
    outputs = tf.keras.layers.Dropout(dropout_rate)(outputs, training=is_training)

    # 残差连接
    outputs += inputs
    
    return outputs
