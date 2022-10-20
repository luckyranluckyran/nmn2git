import math
#from .Init import *
import scipy.spatial
import json
import pickle as pkl
import os
import numpy as np
import tensorflow as tf

from include.Init import *


#输入是一个长度和一个知识图
#输出双向的三元组，以及三元组个数的列表
'''
输入:
    e：知识库内实体的数量
    KG：两个知识库
    注：这里的知识库是{h,r,t}的知识库，利用索引表示，对应于ent_ids_1&2与ref_ent_ids
'''
'''
输出：
    M：一个列表，里面记录了三元组中的双向头尾实体对   
        [(13531, 18721): 1, (18721, 13531): 1, (15978, 13698): 1, (13698, 15978): 1,……]
    du：长度等于两个KG中实体的数量，每一位上的数字，表示以该位数字作为索引的实体在知识库中出现的次数
'''
def get_vmat(e, KG):
    #du,一个列表，长度等于实体数量
    du = [1] * e
    #对知识库中的三元组进行遍历
    for tri in KG:
        #如果头实体与尾实体不相同
        #则在du的对应位置上加一
        if tri[0] != tri[2]:
            du[tri[0]] += 1
            du[tri[2]] += 1
    M = {}
    for tri in KG:
        if tri[0] == tri[2]:
            continue
        if (tri[0], tri[2]) not in M:
            M[(tri[0], tri[2])] = 1
        else:
            pass
        if (tri[2], tri[0]) not in M:
            M[(tri[2], tri[0])] = 1
        else:
            pass

    return M, du

#M表示基于索引的双向KG，e表示KG内元素个数，max_nbr表示邻居数量
def get_nbr(M, e, max_nbr):
    nbr = []
    for i in range(e):
        #在列表nbr后面添加新元素，到后期，nbr长度与e一致
        nbr.append([])
    #对于M中的一对儿
    for (i, j) in M:
        #如果一个头尾实体对不相同，且邻居数为-1（我不理解）或者第i位邻居的长度小于最大邻居数
        if i != j and (max_nbr == -1 or len(nbr[i]) < max_nbr):
            #则在第i位后面填上j
            # 此时生成的列表，每一个位置里面的数字就是对应的邻居索引
            #M = {(1, 4): 1, (4, 1): 1, (5, 8): 1, (8, 5): 1,(2, 7): 1, (7, 2): 1,(9, 2): 1, (2, 9): 1,(6, 3): 1, (3, 6): 1}
            #[[], [4], [7, 9], [6], [1], [8], [3], [2], [5], [2]]
            nbr[i].append(j)
    if max_nbr == -1:
        for i in range(e):
            if (len(nbr[i]) > max_nbr):
                max_nbr = len(nbr[i])
    #输入：
    #nbr=[[4], [], [7, 8], [6], [0], [8], [3], [2], [5, 2], []]，max_nbe = 3
    mask = []
    for i in range(e):
        mask.append([1] * len(nbr[i]) + [0] * (max_nbr - len(nbr[i])))
        nbr[i] += [0] * (max_nbr - len(nbr[i]))
    #输出：
    #nbr = [[4, 0, 0], [0, 0, 0], [7, 8, 0], [6, 0, 0], [0, 0, 0], [8, 0, 0], [3, 0, 0], [2, 0, 0], [5, 2, 0], [0, 0, 0]]
    #mask 可以理解为掩码，将有邻居位置定为1
    #mask = [[1, 0, 0], [0, 0, 0], [1, 1, 0], [1, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 0]]
    return np.asarray(nbr, dtype=np.int32), np.asarray(mask)

#KG实际上就是三元组
'''
输入：
    e:两个知识库实体个数和
    KG：KG1+KG2
'''
'''
输出：
    M0:一个列表，里面记录了三元组中的双向头尾实体对   
        [(13531, 18721): 1, (18721, 13531): 1, (15978, 13698): 1, (13698, 15978): 1,……]
    M:利用稠密矩阵表示M0
'''
def get_sparse_tensor(e, KG):
    print('getting a sparse tensor...')
    '''
    输出：
    M0：一个列表，里面记录了三元组中的双向头尾实体对   
        [(13531, 18721): 1, (18721, 13531): 1, (15978, 13698): 1, (13698, 15978): 1,……]
    du：长度等于两个KG中实体的数量，每一位上的数字，表示以该位数字作为索引的实体在知识库中出现的次数
    '''
    M0, du = get_vmat(e, KG)
    ind = []
    val = []
    for fir, sec in M0:
        #将M0中的实体对序号首末颠倒，存入ind中
        ind.append((sec, fir))
        '''
        将每一个头尾实体对出现的次数（也就是一），除以头实体出现的总次数的平方根后，再除以尾实体出现的总次数的平方根
        再将结果添加至val
        '''
        val.append(M0[(fir, sec)] / math.sqrt(du[fir]) / math.sqrt(du[sec]))

    '''
    用稠密矩阵表示稀疏矩阵，减少空间占用
    1. indices：数据类型为int64的二维Tensor对象，它的Shape为[N, ndims]。indices保存的是非零值的索引，
        即稀疏矩阵中除了indices保存的位置之外，其他位置均为0。
    2. values：一维Tensor对象，其Shape为[N]。它对应的是稀疏矩阵中indices索引位置中的值。
    3. dense_shape：数据类型为int64的一维Tensor对象，其维度为[ndims]，用于指定当前稀疏矩阵对应的Shape

    M=SparseTensor(indices=Tensor("SparseTensor/indices:0", shape=(173130, 2), dtype=int64), 
                    values=Tensor("SparseTensor/values:0", shape=(173130,), dtype=float32), 
                    dense_shape=Tensor("SparseTensor/dense_shape:0", shape=(2,), dtype=int64))
    '''
    M = tf.SparseTensor(indices=ind, values=val, dense_shape=[e, e])

    return M0, M

'''
输入：
    e：两个知识库实体个数和
    dimension： GCN层的隐藏表征的维度是300
    vec_path：训练好的词向量
'''
#根据预先训练好的向量维度，返回
def get_se_input_layer(e, dimension, file_path):
    print('adding the primal input layer...')
    with open(file=file_path, mode='r', encoding='utf-8') as f:
        #嵌入表是vector文件
        embedding_list = json.load(f)
        print(len(embedding_list), 'rows,', len(embedding_list[0]), 'columns.')

    #将词向量转换为tensor
    '''
        type:"tensorflow.python.framework.ops.Tensor"
        value:Tensor("Const:0", shape=(38960, 300), dtype=float32)
    '''
    input_embeddings = tf.convert_to_tensor(embedding_list)
    #tf.Variable()函数用于创建变量(Variable),变量是一个特殊的张量()
    '''
        type:'tensorflow.python.ops.variables.RefVariable'
        value:tf.Variable 'Variable:0' shape=(38960, 300) dtype=float32_ref
    '''
    ent_embeddings = tf.Variable(input_embeddings)

    #利用 L2 范数对词向量进行约束
    return tf.nn.l2_normalize(ent_embeddings, 1)


'''
输入：
    inlayer：input_layer：表示第h层GCN的输出节点，将输入的词向量转化为张量
    dimension：GCN层的隐藏表征的维度是300
    M：利用稠密矩阵表示M0——一个列表，里面记录了三元组中的双向头尾实体对
    act_func：GCN激活函数，对应文章中的等式一
    dropout=0.0：
输出：
    hidden_layer_1：将表示三元组中双向头尾实体对的稠密矩阵，利用GCN进行计算
'''
def add_diag_layer(inlayer, dimension, M, act_func, dropout=0.0, init=ones):
    #将input_layer的神经元随机扔出
    '''
        type:<class 'tensorflow.python.framework.ops.Tensor'>
        value:Tensor("l2_normalize:0", shape=(38960, 300), dtype=float32)
    '''
    inlayer = tf.nn.dropout(inlayer, 1 - dropout)
    print('adding a layer...')
    '''
        type:<class 'tensorflow.python.ops.variables.RefVariable'>
        value:<tf.Variable 'Variable_4:0' shape=(1, 300) dtype=float32_ref>
    '''
    w0 = init([1, dimension])

    #Tensor("Relu:0", shape=(38960, 300), dtype=float32)
    '''
    tf.multiply:两个tensor逐元素相乘
    tf.multiply(inlayer, w0):
        type:<class 'tensorflow.python.framework.ops.Tensor'>
        class:Tensor("Mul_1:0", shape=(38960, 300), dtype=float32) 
    '''
    '''
    tf.sparse_tensor_dense_matmul 是对稀疏张量和稠密张量做矩阵乘法的函数
    '''
    tosum = tf.sparse_tensor_dense_matmul(M, tf.multiply(inlayer, w0))
    if act_func is None:
        return tosum
    else:
        return act_func(tosum)


def highway(layer1, layer2, dimension):
    kernel_gate = glorot([dimension, dimension])
    bias_gate = zeros([dimension])
    transform_gate = tf.matmul(layer1, kernel_gate) + bias_gate
    transform_gate = tf.nn.sigmoid(transform_gate)
    carry_gate = 1.0 - transform_gate
    return transform_gate * layer2 + carry_gate * layer1


def softmax_positiv(T):
    Tsign = tf.greater(T, 0)
    _reduce_sum = tf.reduce_sum(
        tf.exp(tf.where(Tsign, T, tf.zeros_like(T))), -1, keepdims=True) + math.e
    return tf.where(Tsign, tf.exp(T) / _reduce_sum, T)


def neighborhood_matching(inlayer, mask, max_nbr, beta):
    inlayer_ILL = tf.tile(tf.expand_dims(inlayer[0], 2), [1, 1, max_nbr, 1])
    inlayer_can = tf.tile(tf.expand_dims(inlayer[1], 2), [1, 1, max_nbr, 1])
    inlayer_ILL_trans = tf.transpose(inlayer_ILL, [0, 2, 1, 3])
    inlayer_can_trans = tf.transpose(inlayer_can, [0, 2, 1, 3])
    sim_ILL = tf.reduce_sum(tf.multiply(inlayer_ILL, inlayer_can_trans), -1)
    sim_can = tf.reduce_sum(tf.multiply(inlayer_can, inlayer_ILL_trans), -1)
    mask_ILL = tf.expand_dims(mask[0], -1)
    mask_can = tf.expand_dims(mask[1], 1)
    mask_all = tf.einsum('ijk,ikl->ijl', mask_ILL, mask_can)

    a_ILL = softmax_positiv(tf.multiply(sim_ILL, mask_all))
    a_can = softmax_positiv(tf.multiply(
        sim_can, tf.transpose(mask_all, [0, 2, 1])))

    m_ILL = inlayer[0] - tf.reduce_sum(tf.multiply(inlayer_can_trans,tf.expand_dims(a_ILL, -1)), 2)
    m_can = inlayer[1] - tf.reduce_sum(tf.multiply(inlayer_ILL_trans,tf.expand_dims(a_can, -1)), 2)
    #β ∗ mp
    m = tf.stack([m_ILL, m_can], 0) * beta
    #output_layer对应文章中的ˆhp
    output_layer = tf.concat([inlayer, m], -1)
    return output_layer


def mock_neighborhood_matching(inlayer, nbr_weight, max_nbr, beta):
    inlayer_ILL = tf.tile(tf.expand_dims(inlayer[0], 2), [1, 1, max_nbr, 1])
    inlayer_can = tf.tile(tf.expand_dims(inlayer[1], 2), [1, 1, max_nbr, 1])
    inlayer_ILL_trans = tf.transpose(inlayer_ILL, [0, 2, 1, 3])
    inlayer_can_trans = tf.transpose(inlayer_can, [0, 2, 1, 3])
    sim_ILL = tf.reduce_sum(tf.multiply(inlayer_ILL, inlayer_can_trans), -1)
    sim_can = tf.reduce_sum(tf.multiply(inlayer_can, inlayer_ILL_trans), -1)
    weight_ILL = tf.expand_dims(nbr_weight[0], -1)
    weight_can = tf.expand_dims(nbr_weight[1], 1)
    weight_all = tf.einsum('ijk,ikl->ijl', weight_ILL, weight_can)

    a_ILL = softmax_positiv(tf.multiply(sim_ILL, weight_all))
    a_can = softmax_positiv(tf.multiply(
        sim_can, tf.transpose(weight_all, [0, 2, 1])))

    m_ILL = inlayer[0] - \
        tf.reduce_sum(tf.multiply(inlayer_can_trans,
                                  tf.expand_dims(a_ILL, -1)), 2)
    m_can = inlayer[1] - \
        tf.reduce_sum(tf.multiply(inlayer_ILL_trans,
                                  tf.expand_dims(a_can, -1)), 2)
    m = tf.stack([m_ILL, m_can], 0) * beta
    output_layer = tf.concat([inlayer, m], -1)
    return output_layer


def neighborhood_aggregation(outlayer, mask, w_gate, w_N, act_func):
    weight_ij = tf.einsum('ijkl,lp->ijkp', outlayer, w_gate)
    if act_func is not None:
        weight_ij = act_func(weight_ij) 
    h_sum = tf.einsum('ijkl,ijkl->ijkl', outlayer, weight_ij)
    h_sum = tf.reduce_sum(tf.multiply(h_sum, tf.expand_dims(mask, -1)), 2)
    h_j = tf.einsum('ijk,kl->ijl', h_sum, w_N) 
    return h_j


def mock_neighborhood_aggregation(outlayer, nbr_weight, w_gate, w_N, act_func):
    weight_ij = tf.einsum('ijkl,lp->ijkp', outlayer, w_gate)
    if act_func is not None:
        weight_ij = act_func(weight_ij)
    h_sum = tf.einsum('ijkl,ijkl->ijkl', outlayer, weight_ij)
    h_sum = tf.reduce_sum(tf.multiply(
        h_sum, tf.expand_dims(nbr_weight, -1)), 2)
    h_j = tf.einsum('ijk,kl->ijl', h_sum, w_N)
    return h_j


def get_loss_pre(outlayer, ILL, gamma, k, neg_left, neg_right, neg2_left, neg2_right):
    left = ILL[:, 0]
    right = ILL[:, 1]
    left_x = tf.nn.embedding_lookup(outlayer, left)
    right_x = tf.nn.embedding_lookup(outlayer, right)

    A = tf.reduce_sum(tf.abs(left_x - right_x), 1)
    neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
    neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)
    B = tf.reduce_sum(tf.abs(neg_l_x - neg_r_x), 1)
    C = - tf.reshape(B, [-1, k])
    D = A + gamma
    L1 = tf.nn.relu(tf.add(C, tf.reshape(D, [-1, 1])))
    neg_l_x = tf.nn.embedding_lookup(outlayer, neg2_left)
    neg_r_x = tf.nn.embedding_lookup(outlayer, neg2_right)
    B = tf.reduce_sum(tf.abs(neg_l_x - neg_r_x), 1)
    C = - tf.reshape(B, [-1, k])
    L2 = tf.nn.relu(tf.add(C, tf.reshape(D, [-1, 1])))
    # tf.disable_v2_behavior
    # L1 = tf.constant([[1,2,3],[4,5,6],[7,8,9]],dtype=tf.float32)
    # L2 = tf.constant([[9,8,7],[4,5,6],[7,8,9]],dtype=tf.float32)

    #shape:L1,L2:(?, 125)
    #tf.reduce_mean 函数用于计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值
    return (tf.reduce_mean(L1) + tf.reduce_mean(L2)) / 2.0
    # return (L1 + L2) / 2.0


def get_loss_match(outlayer, ILL, gamma, c, dimension):
    out = tf.reshape(outlayer, [2, -1, 2, c, dimension])
    A = tf.reduce_sum(tf.abs(out[0, :, 0, 0] - out[1, :, 0, 0]), -1)
    B = tf.reduce_sum(tf.abs(out[0, :, 0, 1:c] - out[1, :, 0, 1:c]), -1)
    C = - tf.reshape(B, [-1, c - 1])
    D = A + gamma
    L1 = tf.nn.relu(tf.add(C, tf.reshape(D, [-1, 1])))
    B = tf.reduce_sum(tf.abs(out[0, :, 1, 1:c] - out[1, :, 1, 1:c]), -1)
    C = - tf.reshape(B, [-1, c - 1])
    L2 = tf.nn.relu(tf.add(C, tf.reshape(D, [-1, 1])))
    return (tf.reduce_mean(L1) + tf.reduce_mean(L2)) / 2.0


def get_loss_w(select_train, outlayer, nbr_all,
                    mask_all, sample_w, w_gate, w_N,
                    ILL, max_nbr_all, beta):
    left = tf.gather(ILL[:, 0], select_train)
    right = tf.gather(ILL[:, 1], select_train)
    t = 10
    idx = tf.concat([left, right], axis=0) 

    outlayer_idx = tf.gather(outlayer, idx)
    nbr_idx = tf.gather(nbr_all, idx)
    mask_idx = tf.to_float(tf.gather(mask_all, idx))
    outlayer_nbr_idx = tf.gather(outlayer, nbr_idx)
    out_sim = tf.einsum('ij,ijk->ik', tf.matmul(outlayer_idx, sample_w),
                        tf.transpose(outlayer_nbr_idx, [0, 2, 1]))
    nbr_weight = tf.reshape(softmax_positiv(tf.multiply(
        out_sim, mask_idx)), (2, t, -1)) 

    outlayer_idx = tf.reshape(outlayer_idx, (2, t, -1))
    nbr_idx = tf.reshape(nbr_idx, (2, t, -1))
    outlayer_nbr_idx = tf.gather(outlayer, nbr_idx)
    mock_hat_h = mock_neighborhood_matching(
        outlayer_nbr_idx, nbr_weight, max_nbr_all, beta) 
    mock_g = mock_neighborhood_aggregation(
        mock_hat_h, nbr_weight, w_gate, w_N, tf.sigmoid)
    left_x = tf.concat([outlayer_idx[0], mock_g[0]], axis=-1)
    right_x = tf.concat([outlayer_idx[1], mock_g[1]], axis=-1)

    A = tf.reduce_sum(tf.abs(left_x - right_x), 1)
    return tf.reduce_mean(A)

'''
输入：
    dimension：config.dim：GCN层的隐藏表征的维度是300
    dimension_g：config.dim_g:邻域表征的维度是50
    act_func：config.act_func: GCN激活函数，对应文章中的等式一
    gamma：config.gamma:公式10中的γ
    k：config.k:消极对的数量
    vec_path：config.vec: 训练好的词向量
    e：两个知识库实体个数和
    all_nbr_num：config.all_nbr_num：所有可以被涉及的邻居的数量 
    sampled_nbr_num：config.sampled_nbr_num：被采样的邻居的数量 
    beta：config.beta：β，论文中公式（6）涉及的参数，加权匹配向量的参数。
    KG：KG1 + KG2：源知识库与目标知识库的结合
'''
'''
输出：
    output_h：
    output_h_match：
    loss_all ：
    sample_w：
    loss_w：
    M0：
    nbr_all： 
    mask_all：
'''
def build(dimension, dimension_g, act_func, gamma, k, vec_path, e, all_nbr_num, sampled_nbr_num, beta, KG):
    #对图进行重置，仅对当前线程生效
    tf.reset_default_graph()
    '''
    输入：
        e：两个知识库实体个数和
        dimension： GCN层的隐藏表征的维度是300
        vec_path：训练好的词向量
    '''
    '''
    输出：
        将输入的词向量转化为张量
    '''
    #对应论文中的公式9，表示第h层GCN的输出节点
    #input_layer:type：'tensorflow.python.framework.ops.Tensor'
    #            value:Tensor("l2_normalize_1:0", shape=(38960, 300), dtype=float32)
    input_layer = get_se_input_layer(e, dimension, vec_path)

    #M0:dict类型,一个基于索引对应的双向字典
    #M0：基于索引的双向KG
    #M0:{(23621, 8178): 1, (8178, 23621): 1, (5598, 5837): 1, (5837, 5598): 1,……
    #M0：
    #M=SparseTensor(indices=Tensor("SparseTensor/indices:0", shape=(26566, 2), dtype=int64),
    #               values=Tensor("SparseTensor/values:0", shape=(26566,),dtype=float32),
    #               dense_shape=Tensor("SparseTensor/dense_shape:0", shape=(2,), dtype=int64))
    '''
    输入：
        e:两个知识库实体个数和
        KG：KG1+KG2
    '''
    '''
    输出：
        M0:一个列表，里面记录了三元组中的双向头尾实体对   
            [(13531, 18721): 1, (18721, 13531): 1, (15978, 13698): 1, (13698, 15978): 1,……]
        M:利用稠密矩阵表示M0
        M0：tensorflow.python.framework.sparse_tensor.SparseTensor类型（这是一个奇葩的我也不知道的类型）
                indices：数据类型为int64的二维Tensor对象，它的Shape为[N, ndims]。保存的是非零值的索引，即稀疏矩阵中除了indices保存的位置之外，其他位置均为0
                values：一维Tensor对象，其Shape为[N]。它对应的是稀疏矩阵中indices索引位置中的值。
                dense_shape：数据类型为int64的一维Tensor对象，其维度为[ndims]，用于指定当前稀疏矩阵对应的Shape。
        M：SparseTensor(indices=Tensor("SparseTensor/indices:0", shape=(26566, 2), dtype=int64),
                values：Tensor("SparseTensor/values:0", shape=(26566,),dtype=float32),
                dense_shape：Tensor("SparseTensor/dense_shape:0", shape=(2,), dtype=int64))
    '''
    M0, M = get_sparse_tensor(e, KG)
    '''
    输入：
        M0：表示基于索引的双向KG
        e：表示KG内元素个数
        all_nbr_num：表示邻居数量
    '''
    '''
    输出：
        nbr_all = [[4, 0, 0], [0, 0, 0], [7, 8, 0], [6, 0, 0], [0, 0, 0], [8, 0, 0], [3, 0, 0], [2, 0, 0], [5, 2, 0], [0, 0, 0]]
        mask 可以理解为掩码，将有邻居位置定为1
        mask_all = [[1, 0, 0], [0, 0, 0], [1, 1, 0], [1, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 0]]
    '''

    nbr_all, mask_all = get_nbr(M0, e, all_nbr_num)

    print('KG structure embedding')
    '''
    输入：
        input_layer：表示第h层GCN的输出节点，将输入的词向量转化为张量
        dimension：GCN层的隐藏表征的维度是300
        M：利用稠密矩阵表示M0——一个列表，里面记录了三元组中的双向头尾实体对
        act_func：GCN激活函数，对应文章中的等式一
        dropout=0.0：
    输出：
        hidden_layer_1：将表示三元组中双向头尾实体对的稠密矩阵，利用GCN进行计算
    '''
    hidden_layer_1 = add_diag_layer(input_layer, dimension, M, act_func, dropout=0.0)
    '''
    highway网络控制噪声
    '''
    hidden_layer = highway(input_layer, hidden_layer_1, dimension)
    '''
        再次进行GCN网络计算
    '''
    hidden_layer_2 = add_diag_layer(hidden_layer, dimension, M, act_func, dropout=0.0)
    '''
        再次利用highway控制噪声
        注意：output_h是GCN网络的输出层
    '''
    '''
    output_h:两个参数，第一个参数38900是源知识库加上目标知识库中每一对三元组，双向存储的个数，第二个参数300是GCN的层
        shape:(38960, 300)
        type:<class 'tensorflow.python.framework.ops.Tensor'>
        value:Tensor("add_3:0", shape=(38960, 300), dtype=float32)
    '''
    output_h = highway(hidden_layer, hidden_layer_2, dimension)
    print('shape of output_h: ', output_h.get_shape())
    '''  
        tf.placeholder(dtype,shape=None,name=None)
            dtype：数据类型。常用的是tf.float32,tf.float64等数值类型
            shape：数据形状。默认是None，就是一维值，也可以是多维（比如[2,3], [None, 3]表示列是3，行不定）
            name：名称
        所以placeholder()函数是在神经网络构建graph的时候在模型中的占位，此时并没有把要输入的数据传入模型，
        它只会分配必要的内存。等建立session，在会话中，运行模型的时候通过feed_dict()函数向占位符喂入数据。
    '''
    '''
    以下五行定义了五个神经网络graph
    '''
    '''
        type: <class 'tensorflow.python.framework.ops.Tensor'>
        shape: <unknown>
        value: Tensor("c:0", dtype=int32)
    '''
    c = tf.placeholder(tf.int32, None, "c")
    '''
        type: <class 'tensorflow.python.framework.ops.Tensor'>
        shape: (38960, 3)
        value: Tensor("nbr_sampled:0", shape=(38960, 3), dtype=int32)
    '''
    nbr_sampled = tf.placeholder(tf.int32, [e, sampled_nbr_num], "nbr_sampled")
    '''
        type: <class 'tensorflow.python.framework.ops.Tensor'>
        shape: (38960, 3)
        value: Tensor("mask_sampled:0", shape=(38960, 3), dtype=float32)
    '''
    mask_sampled = tf.placeholder(tf.float32, [e, sampled_nbr_num], "mask_sampled")
    '''
        type: <class 'tensorflow.python.framework.ops.Tensor'>
        shape: (?, 2)
        value: Tensor("ILL:0", shape=(?, 2), dtype=int32)   
    '''
    ILL = tf.placeholder(tf.int32, [None, 2], "ILL")
    '''
        type: <class 'tensorflow.python.framework.ops.Tensor'>
        shape: (?,)
        value: Tensor("candidate:0", shape=(?,), dtype=int32)
    '''
    candidate = tf.placeholder(tf.int32, [None], "candidate")

    '''
    tf.reshape(tensor, shape, name=None)
        tensor:需要被reshape的张量
        shape：新形状应与原始形状兼容。如果是整数，则结果将是该长度的一维数组。一个形状尺寸可以为-1。
                在这种情况下，该值是根据数组的长度和其余维来推断的。
    tf.transpose(input, [dimension_1, dimenaion_2,..,dimension_n])
        input:待转换的张量
        [dimension_1, dimenaion_2,..,dimension_n]：交换的维度以及位置
    tf.expand_dims(input, axis=None, name=None, dim=None)
        input是输入张量。
        axis是指定扩大输入张量形状的维度索引值。
        dim等同于轴，一般不推荐使用。
    tf.tile(input,multiples,name=None)，详阅：https://blog.csdn.net/tsyccnh/article/details/82459859
        input：待扩展的张量
        multiples：扩展的方法
        name=None：
    tf.stack(values,axis=0,name=’’)详阅：https://blog.csdn.net/william_hehe/article/details/78889631
        values：两个张量按照指定的方向进行叠加，生成一个新的张量。
        axis=0
    tf.gather(params, indices),详阅：https://zhuanlan.zhihu.com/p/495164112
        params：待切片的张量
        indices：切片的依据——索引
    tf.nn.embedding_lookup(params,ids,partition_strategy='mod',name=None,validate_indices=True,max_norm=None)
        params: 表示完整的嵌入张量，或者除了第一维度之外具有相同形状的P个张量的列表，表示经分割的嵌入张量
        ids: 一个类型为int32或int64的Tensor，包含要在params中查找的id
        partition_strategy: 指定分区策略的字符串，如果len（params）> 1，则相关。当前支持“div”和“mod”。 默认为“mod”
        name: 操作名称（可选）
        validate_indices:  是否验证收集索引
        max_norm: 如果不是None，嵌入值将被l2归一化为max_norm的值
    '''
    #候选实体
    '''
    type:<class 'tensorflow.python.framework.ops.Tensor'>
    shape:(?,)
    value:Tensor("Reshape_1:0", shape=(?,), dtype=int32)
    '''
    candidate = tf.reshape(tf.transpose(tf.reshape(candidate, (2, -1, c)), (1, 0, 2)), [-1])
    #种子对和候选实体结合
    '''
    type: <class 'tensorflow.python.framework.ops.Tensor'>
    shape: (2, ?)
    value: Tensor("stack:0", shape=(2, ?), dtype=int32)
    '''
    idx_pair = tf.stack([tf.reshape(tf.tile(tf.expand_dims(ILL, -1), (1, 1, c)), [-1]), candidate])
    #邻居对
    '''
    type: <class 'tensorflow.python.framework.ops.Tensor'>
    shape: (2, ?, 3)
    value: Tensor("GatherV2:0", shape=(2, ?, 3), dtype=int32)
    '''
    nbr_pair = tf.gather(nbr_sampled, idx_pair)
    #掩码对
    '''
    type: <class 'tensorflow.python.framework.ops.Tensor'>
    shape: (2, ?, 3)
    value: Tensor("GatherV2_1:0", shape=(2, ?, 3), dtype=float32)
    '''
    mask_pair = tf.gather(mask_sampled, idx_pair)
    '''
    type: <class 'tensorflow.python.framework.ops.Tensor'>
    shape: (2, ?, 300)
    value: Tensor("embedding_lookup/Identity:0", shape=(2, ?, 300), dtype=float32)
    '''
    h_ctr = tf.nn.embedding_lookup(output_h, idx_pair)
    '''
    type: <class 'tensorflow.python.framework.ops.Tensor'>
    shape: (2, ?, 3, 300)
    value: Tensor("embedding_lookup_1/Identity:0", shape=(2, ?, 3, 300), dtype=float32)
    '''
    h_nbr = tf.nn.embedding_lookup(output_h, nbr_pair)

    #neighborhood aggregation中涉及的两个参数，在维度表征部分得以体现
    '''
        <class 'tensorflow.python.ops.variables.RefVariable'>
        <tf.Variable 'Variable_7:0' shape=(600, 600) dtype=float32_ref>
    '''
    w_gate = glorot([dimension * 2, dimension * 2])
    '''
        <class 'tensorflow.python.ops.variables.RefVariable'>
        <tf.Variable 'Variable_8:0' shape=(600, 50) dtype=float32_ref>
    '''
    w_N = glorot([dimension * 2, dimension_g])


    #邻域匹配
    print('neighborhood matching')
    output_hat_h = neighborhood_matching(h_nbr, mask_pair, sampled_nbr_num, beta)
    #shape of output_hat_h:  (2, ?, 3, 600)
    print('shape of output_hat_h: ', output_hat_h.get_shape())

    #邻域聚集
    print('neighborhood aggregation')
    output_g = neighborhood_aggregation(
        output_hat_h, mask_pair, w_gate, w_N, tf.sigmoid)
    output_h_match = tf.concat([tf.reshape(
        h_ctr, [-1, dimension]), tf.reshape(output_g, [-1, dimension_g])], -1)
    dimension3 = dimension + dimension_g
    #shape of output_h_match:  (?, 350)
    print('shape of output_h_match: ', output_h_match.get_shape())

    #预训练损失函数的计算
    print("compute pre-training loss")
    neg_left = tf.placeholder(tf.int32, [None], "neg_left") 
    neg_right = tf.placeholder(tf.int32, [None], "neg_right")
    neg2_left = tf.placeholder(tf.int32, [None], "neg2_left")
    neg2_right = tf.placeholder(tf.int32, [None], "neg2_right")
    #对应论文中的公式10
    loss_pre = get_loss_pre(output_h, ILL, gamma, k, neg_left,
                      neg_right, neg2_left, neg2_right)
    # tf.disable_v2_behavior
    # tensor = tf.constant([[1,2,3],[4,5,6],[7,8,9]],dtype=tf.float32)
    # print(tensor)
    # array = tf.Session().run(tensor)
    # print(array)
    #print("loss_pre")
    #print(loss_pre)
    # tf.disable_v2_behavior
    # loss_pre = tf.constant([[1,2,3],[4,5,6],[7,8,9]],dtype=tf.float32)
    #print(loss_pre.shape)
    # zero_array = np.zeros(loss_pre.shape)
    '''
    array1 = loss_pre.eval(session=tf.Session())
    print("array1")
    print(array1)
    '''
    '''
        <class 'tensorflow.python.framework.ops.Tensor'>
        Tensor("truediv_2:0", shape=(), dtype=float32)
    '''




    # 整体损失函数的计算
    print("compute overall loss")
    loss_match = get_loss_match(output_h_match, ILL, gamma, c, dimension3)
    alpha = tf.placeholder(tf.float32, None, "alpha")
    loss_all = (1 - alpha) * loss_pre + alpha * loss_match
    '''
        <class 'tensorflow.python.framework.ops.Tensor'>
        Tensor("add_14:0", dtype=float32)
    '''



    # 采样损失函数的计算
    print("compute sampling process loss")
    select_train = tf.placeholder(tf.int32, [10], "select_train")
    #对应论文中的公式13，负责聚合采样的所有邻居信息
    sample_w = tf.Variable(tf.eye(dimension, name="sample_w"))
    #对应论文中的公式14，目标是训练Ws，Ws是用来训练一跳邻居与中心实体相似程度的权重矩阵
    loss_w = get_loss_w(select_train, output_h, nbr_all,
                             mask_all, sample_w, w_gate, w_N,
                             ILL, all_nbr_num, beta)
    '''
        <class 'tensorflow.python.framework.ops.Tensor'>
        Tensor("Mean_4:0", shape=(), dtype=float32)
    '''


    return output_h, output_h_match, loss_all, sample_w, loss_w, M0, nbr_all, mask_all


'''
输入：
    ILL:ILL[:, 1]：右边的种子实体对
    output_layer:out：根据GCN网络的输出层起一个线程
    k：消极对的数量
    batchnum:train_batchnum：训练集邻域采样数量 
'''
'''
输出：
    
'''
def get_neg(ILL, output_layer, k, batchnum):
    neg = []
    #种子对内实体数量
    t = len(ILL)
    '''
    ILL_vec:源知识库内实体的GCN输出
        shape:(4500, 300)
        value:[[-0.01945713 -0.03095501  0.00057604 ... -0.0198629   0.02137855  0.00408904]
                [ 0.00548146 -0.01594982 -0.00812281 ...  0.03700393  0.00749667  0.02632143]
                [-0.00401904 -0.02044861  0.00238539 ...  0.00600081 -0.00133099 -0.00243945]
                ...
                [ 0.02775401 -0.01436116 -0.00351051 ...  0.01331489  0.00996299 -0.01629847]
                [-0.02124756  0.00702052  0.03480983 ... -0.01143852  0.05344791 -0.00786267]
                [ 0.00298746  0.0216015  -0.01902119 ...  0.01370443  0.01733686 0.00033913]]
    '''
    ILL_vec = np.array([output_layer[e1] for e1 in ILL])
    '''
    KG_vec：整体的GCN输出
        shape:(38960, 300)
        value:[[-0.01366728 -0.01286478  0.01497454 ...  0.00238507  0.03369818 -0.00381335]
                [-0.00340466  0.01463187 -0.00622568 ...  0.02353746  0.03872197 0.02477079]
                [ 0.00342567  0.01258907  0.00232032 ... -0.00338298  0.02242678 0.00219307]
                ...
                [ 0.03314352 -0.00944472  0.02667018 ...  0.00144972  0.01259718 0.03286751]
                [-0.01053051 -0.01623724  0.0617447  ...  0.00287061  0.03093156 0.02632413]
                [-0.00496356  0.00677516 -0.01788223 ... -0.00392106  0.0014925 0.00458268]]
    '''
    KG_vec = np.array(output_layer)
    for p in range(batchnum):
        head = int(t / batchnum * p)

        if p==batchnum-1:
            tail=t
        else:
            tail = int(t / batchnum * (p + 1))

        '''
        scipy.spatial.distance.cdist(XA, XB, metric='euclidean', p=None, V=None, VI=None, w=None)，
        该函数用于计算两个输入集合的距离，通过metric参数指定计算距离的不同方式得到不同的距离度量值
        '''
        #曼哈顿距离
        '''
        ILL_vec[head:tail]
            shape:(4500, 300)
            type:numpy.ndarray
            value:[[-0.01911003 -0.03010087 -0.00338014 ... -0.01956166  0.01589996  0.00408359]
                   [ 0.00308578 -0.01569899 -0.00579189 ...  0.03265805  0.00753824  0.01520988]
                   [-0.01108812 -0.00900319 -0.03608275 ...  0.01651247  0.00409352 -0.01702861]
                    ...
                   [ 0.01102888 -0.01283005  0.00420539 ...  0.01446933  0.0085097  -0.01447012]
                   [-0.02200334  0.00557747  0.02539237 ... -0.01147882  0.04874773 -0.00776372]
                   [-0.01249978  0.00393807  0.00592835 ...  0.01557181  0.01931917  0.01853033]]
        sim:测量源知识库中实体与全部实体嵌入的曼哈顿距离
            shape:(4500, 38960)
            type:numpy.ndarray
            value:[[5.8540423  6.7058345  5.16173495 ... 5.7468296  6.45596341 5.53738476]
                   [5.25272397 5.38001262 4.88411794 ... 5.38041198 5.75962016 4.55185871]
                   [5.01461506 5.52277401 4.34653563 ... 5.454493   5.67044012 4.7951595 ]
                   ...
                   [4.81979479 5.59128142 4.74935788 ... 5.74704272 5.66438978 5.2459146 ]
                   [4.52387589 6.09711507 4.99000049 ... 5.1045658  5.51656127 5.34186508]
                   [4.63972219 3.78522874 4.45747972 ... 5.28722644 5.16393875 4.69141192]]
        '''
        sim = scipy.spatial.distance.cdist(ILL_vec[head:tail], KG_vec, metric='cityblock')

        for i in range(tail - head):
            '''
            argsort()
            功能: 将矩阵a按照axis排序，并返回排序后的下标
            参数: a:输入矩阵
                 axis:需要排序的维度
                 返回值: 输出排序后的下标
            '''
            '''
            sim[i, :]
                type:<class 'numpy.ndarray'>
                shape:(38960,)
                value:[6.69879875 7.3947263  6.16465859 ... 6.57045683 7.32131915 6.34642205]
            '''
            '''
            rank：sim[i, :]进行排序，返回排序后元素的下标
                type:<class 'numpy.ndarray'>
                shape:(38960,)
                value:[37240 13124 22129 ... 13614 34630 34479]
            '''
            rank = sim[i, :].argsort()

            '''
            rank[0: k]
                <class 'numpy.ndarray'>
                (125,)
                [37240 13124 22129 16162 10842  2624 18718   342  8218 15442 36435 16727
                 15784 11362 35402  4735 36355 17064  5662 16312 13191 17729 23082 16262
                 37878 32564 12573 15908  5812 29325  6227 21131 18011  7229 19689 16884
                 13962 36669 19106  6971 13402  8899 23966  4942  1018 31542 17471  7511
                  8606  9189 32816 33006 26610 19980 33660 28967 19399 13418  6384  6564
                  2902 30405 32766 31208 10717 18303 25756 20450  9480 14093 10740 16557
                 32543 13666 27618 18513 26906 38570 25744    22  9950 28552  5408 29422
                 30025   929  3593 11429 12485  4752 16538  5284 16740  8013 31914 35214
                  2918 35700 36130  7637  2073 32433 11597  5762  6057 12171   240 13736
                   862 24013 38309 17964 12416 19372 35412  3722 15959 27173 31319 17425
                 33538 16242 26131  6038  5568]
            '''
            #只将排名的前k个值纳入neg中，排在前面的相对不相关
            neg.append(rank[0: k])

    '''
    neg:函数的输入是个；list

    neg
        type:numpy.ndarray
        shape:(4500, 125)
        value:[[37240 13124 22129 ...  6038 26131  5568]
               [16446  5946  2546 ... 37594 12661 16904]
               [12949 31117 38576 ...  8468  9790  1894]
               ...
               [36364 21029 22300 ...  3297  6754 17083]
               [35687 21047 12813 ... 21182 14991 33225]
               [18035  9678 38515 ... 11193 15385 13799]]
    '''
    neg = np.array(neg)
    '''
    neg
        type:numpy.ndarray
        shape:(562500,)
        value:[37240 13124 22129 ... 11193 15385 13799]
    '''
    neg = neg.reshape((t * k,))

    return neg


def np_softmax(x, T=0.1):
    x = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(x / T) / np.sum(np.exp(x / T), axis=-1, keepdims=True)


'''
输入：
    out：GCN输出，数量等于两个知识库内所有双向实体对的个数，(38960, 300)
    nbr_all：每一个节点对应的邻居的编码
    mask_all：nbr_all的掩码——nbr_all非空位置1
    e：两个知识库实体个数和
    max_nbr：sampled_nbr_num：邻居的数量
    w:sample_w_vec：返回一个维度为dimension【GCN层的隐藏表征的维度：300】的对角阵（轴线为1，其余位为0）
    batchnum:test_batchnum：测试集邻域采样数量
输出：
    nbr:根据注意力权重，从nbr_all中随机抽取邻居
    mask:nbr的掩码矩阵
'''
def sample_nbr(out, nbr_all, mask_all, e, max_nbr, w, batchnum):
    nbr = []
    for p in range(batchnum):
        head = int(e / batchnum * p)
        if p==batchnum-1:
            tail=e
        else:
            tail = int(e / batchnum * (p + 1))
        #mask_p与nbr_p是根据head和tail从mask_all和nbr_all中抽出的内容
        mask_p = mask_all[head:tail]
        nbr_p = nbr_all[head:tail]
        '''
        np.dot()函数主要有两个功能，向量点积和矩阵乘法
        '''
        #sim:GCN输出矩阵乘对角阵，并转秩
        '''
        sim
            shape:(7792, 38960)
            value:[[ 0.08122684  0.02442035  0.00786117 ...  0.01394351  0.03227489  0.00281551]
                   [ 0.02442035  0.15978521  0.02478387 ...  0.03413255  0.0243585   0.04509957]
                   [ 0.00786117  0.02478387  0.06281491 ...  0.03877179  0.01190887  0.03201725]
                    ...
                   [-0.00545705 -0.0023538  -0.0056094  ... -0.00355933 -0.00972454  0.00333368]
                   [ 0.01167345  0.01693149 -0.0046218  ... -0.00162202  0.00605182 -0.00060946]
                   [ 0.00166695 -0.00815105 -0.00252354 ... -0.00644967 -0.00221965 -0.00921833]]
        '''
        sim = np.dot(np.dot(out[head:tail], w), out.transpose())
        #numpy.tile()：把数组沿各个方向复制
        #np.arange()返回一个有终点和起点的固定步长的排列，如[1,2,3,4,5]，起点是1，终点是6，步长为1
        '''
        np.arange(tail - head):返回一个长度为tail - head，步长为一的数组
        np.tile(np.arange(tail - head),(nbr_all.shape[1], 1))：将上述数组拓展为长度为nbr_all行数的数组
        x_axis_index：将上述数组转秩
        '''
        '''
        np.arange(tail - head)
            shape:(7792,)
            value:[   0    1    2 ... 7789 7790 7791]
        
        np.tile(np.arange(tail - head),(nbr_all.shape[1], 1))
            shape:(100, 7792)
            value:[[   0    1    2 ... 7789 7790 7791]
                   [   0    1    2 ... 7789 7790 7791]
                   ...
                   [   0    1    2 ... 7789 7790 7791]
                   [   0    1    2 ... 7789 7790 7791]]

        x_axis_index
            shape:(7792, 100)
            value:[[   0    0    0 ...    0    0    0]
                   [   1    1    1 ...    1    1    1]
                   ...
                   [7790 7790 7790 ... 7790 7790 7790]
                   [7791 7791 7791 ... 7791 7791 7791]]
        '''
        x_axis_index = np.tile(np.arange(tail - head),(nbr_all.shape[1], 1)).transpose()

        eps = 1e-8
        '''
        sim[x_axis_index, nbr_p]
            shape:(7792, 100)
            value:[[ 0.05970707  0.04925476  0.08122684 ...  0.08122684  0.08122684  0.08122684]
                   [ 0.14747043  0.02442035  0.02442035 ...  0.02442035  0.02442035  0.02442035]
                   ...
                   [ 0.03361882  0.06376377  0.01167345 ...  0.01167345  0.01167345  0.01167345]
                   [ 0.00166695  0.00166695  0.00166695 ...  0.00166695  0.00166695  0.00166695]]
        prob1
            shape:(7792, 100)
            value:[[ 5.97070736e-02  4.92547577e-02 -9.99999999e+07 ... -9.99999999e+07 -9.99999999e+07 -9.99999999e+07]
                   [ 1.47470431e-01 -1.00000000e+08 -1.00000000e+08 ... -1.00000000e+08 -1.00000000e+08 -1.00000000e+08]
                   ...
                   [ 3.36188167e-02  6.37637728e-02 -1.00000000e+08 ... -1.00000000e+08 -1.00000000e+08 -1.00000000e+08]
                   [-1.00000000e+08 -1.00000000e+08 -1.00000000e+08 ... -1.00000000e+08 -1.00000000e+08 -1.00000000e+08]]
        '''
        prob = sim[x_axis_index, nbr_p] - 1e8 * (1 - mask_p)
        '''
        prob2
            shape:(7792, 100)
            value:[[0.52610704 0.47389298 0.         ... 0.         0.         0.        ]
                   [1.00000001 0.         0.         ... 0.         0.         0.        ]
                   [0.01       0.01       0.01       ... 0.01       0.01       0.01      ]
                   ...
                   [0.01       0.01       0.01       ... 0.01       0.01       0.01      ]
                   [0.42520317 0.57479685 0.         ... 0.         0.         0.        ]
                   [0.01       0.01       0.01       ... 0.01       0.01       0.01      ]]
        '''
        prob = np_softmax(prob) + eps * mask_p

        '''
        prob3
            shape:(7792, 100)
            value:[[0.52610703 0.47389297 0.         ... 0.         0.         0.        ]
                   [1.         0.         0.         ... 0.         0.         0.        ]
                   [0.01       0.01       0.01       ... 0.01       0.01       0.01      ]
                   ...
                   [0.01       0.01       0.01       ... 0.01       0.01       0.01      ]
                   [0.42520317 0.57479683 0.         ... 0.         0.         0.        ]
                   [0.01       0.01       0.01       ... 0.01       0.01       0.01      ]]
        '''

        #对应论文中的公式4，表示注意力权重
        prob = prob / np.sum(prob, axis=1, keepdims=True)



        for i in range(tail - head):
            if np.sum(mask_p[i]) > max_nbr:
                nbr.append(nbr_p[i, np.random.choice(
                    nbr_all.shape[1], max_nbr, replace=False, p=prob[i])])
            else:
                nbr.append(nbr_p[i, 0:max_nbr])
    #mask_all的最大邻居数
    mask = mask_all[:, 0:max_nbr]

    return nbr, mask

'''
输入：
    e：两个知识库内实体的总数
    e1：源知识库
    e2：目标知识库
'''
'''
输出：
    mask_e1：长度等于e，e1中的实体对应的索引位置1
    mask_e2：长度等于e，e2中的实体对应的索引位置1
'''
def mask_candidate(e, e1, e2):
    mask_e1 = np.zeros(e)
    mask_e2 = np.zeros(e)
    for x in e1:
        mask_e1[x] = 1
    for x in e2:
        mask_e2[x] = 1
    return mask_e1, mask_e2


'''
输入:
    ILL:ILL[:, 1]:种子对右边的实体
    ILL_true：ILL[:, 0]:种子对左边的实体
    out:以数组的形式表示GCN的输出层，其数量等于两个知识库内所有的双向三元组
    k：c:每个实体的候选集个数
    mask_e：mask_e2:长度等于e，e2中的实体对应的索引位,置1
    batchnum：train_batchnum:训练集邻域采样数量
输出:
    c_left:
'''
def sample_candidate(ILL, ILL_true, out, k, mask_e, batchnum):
    '''
    t：种子对右边实体的个数
    e：GCN输出层参数的个数，也就是所有双向二元组的个数
    '''
    t = len(ILL)
    e = len(out)
    #筛选出种子对右边的实体对应的GCN输出层
    ILL_vec = np.array([out[x] for x in ILL])
    #GCN输出层
    KG_vec = np.array(out)
    neg = []
    for p in range(batchnum):
        head = int(t / batchnum * p)
        if p==batchnum-1:
            tail=t
        else:
            tail = int(t / batchnum * (p + 1))
        #种子对右边的实体与全部实体的曼哈顿距离
        sim = scipy.spatial.distance.cdist(ILL_vec[head:tail], KG_vec, metric='cityblock')
        #定义一个矩阵，横纵尺寸分别为：一批样本的数量和实体的总个数
        mask_gold = np.zeros((tail - head, e))
        for i in range(tail - head):
            #出现在种子对左边的实体的索引位置1
            mask_gold[i][ILL_true[i + head]] = 1
        '''
        mask_gold:注：value中肯定有1，只是没有显示而已
            shape:(4500, 38960)
            value:[[0. 0. 0. ... 0. 0. 0.]
                   [0. 0. 0. ... 0. 0. 0.]
                   [0. 0. 0. ... 0. 0. 0.]
                   ...
                   [0. 0. 0. ... 0. 0. 0.]
                   [0. 0. 0. ... 0. 0. 0.]
                   [0. 0. 0. ... 0. 0. 0.]]
        '''
        mask = np.tile(mask_e, (tail - head, 1)) + mask_gold
        '''
        - sim - 1e8 * mask_gold
            shape:(4500, 38960)
            type:[[-6.70381679 -7.40288893 -6.15900686 ... -6.55481024 -7.30420631 -6.35029543]
                  [-5.51306793 -5.5400375  -5.33552778 ... -5.64734236 -5.99412302 -4.8427256 ]
                  [-6.0677241  -6.34426631 -5.27992792 ... -5.86148163 -6.07764249 -5.50970441]
                  ...
                  [-5.2422612  -5.82943988 -5.19589895 ... -5.95683927 -6.02955847 -5.48168209]
                  [-4.52335104 -6.10184524 -4.98770625 ... -5.10384062 -5.4735625  -5.33560783]
                  [-5.41456639 -5.75610743 -5.15544024 ... -5.37735832 -5.99868894 -4.77180226]]

        prob:（- sim - 1e8 * mask_gold）的归一化
            shape:(4500, 38960)
            value:[[7.68583527e-30 7.07390691e-33 1.78554394e-27 ... 3.41050203e-29 1.89772168e-32 2.63642147e-28]
                   [1.14038155e-24 8.70808895e-25 6.73127382e-24 ... 2.97785658e-25 9.28655165e-27 9.29629625e-22]
                   ...
                   [2.26639725e-20 3.16244401e-27 2.18105674e-22 ... 6.82813047e-23 1.69285628e-24 6.72588786e-24]
                   [3.05377439e-24 1.00355939e-25 4.07574968e-23 ... 4.43026328e-24 8.87207004e-27 1.88940955e-21]]
        '''
        prob = np_softmax(- sim - 1e8 * mask_gold)

        for i in range(tail - head):
            '''
            numpy.random.choice(a, size=None, replace=True, p=None)
                a:从a(只要是ndarray都可以，但必须是一维的)中随机抽取数字
                size:组成指定大小(size)的数组
                replace:True表示可以取相同数字，False表示不可以取相同数字
                p：与数组a相对应，表示取数组a中每个元素的概率，默认为选取每个元素的概率相同。
            '''
            #从全体实体中随机抽取实体加入消极对，被抽中的概率和归一化的prob中的概率成正比
            #出现在种子对左边的次数越多，被抽中的概率越大
            neg.append(np.random.choice(e, k - 1, replace=False, p=prob[i]))
    #np.concatenate()是用来对数列或矩阵进行合并的
    candidate = np.concatenate((np.expand_dims(ILL_true, -1), np.asarray(neg)), axis=1)
    #只选择前k个，k为候选实体个数
    candidate = candidate.reshape((t * k,))
    '''
    candidate
        shape:(90000,)
        type:numpy.ndarray
    '''
    return candidate

'''
输入：
    output_h:GCN网络的输出层
    output_h_match:邻域聚集阶段的输出层
    loss_all:整体的损失函数
    sample_w:聚合采样的所有的邻居信息
    loss_w:训练Ws的损失函数
    learning_rate：config.lr:学习率
    epochs：config.epochs:训练批次
    pre_epochs：config.pre_epochs:预训练（也就是GCN）的训练批次
    ILL：train:30%的训练集
    e:两个知识库实体个数和
    k：config.k:消极对的数量
    sampled_nbr_num：config.sampled_nbr_num:邻居的数量
    save_suffix：config.save_suffix:数据集与语言：DBP15k_zh_en
    dimension：config.dim:GCN层的隐藏表征的维度是300
    dimension_g：config.dim_g:邻域表征的维度是50
    c：config.c:每个实体的候选集个数
    train_batchnum：config.train_batchnum:训练集邻域采样数量 
    test_batchnum：config.test_batchnum:测试集邻域采样数量
    test:70%的测试集
    M0:表示基于索引的双向KG
    e1:源知识库
    e2:目标知识库
    nbr_all:每一个节点对应的邻居的编码
    mask_all:nbr_all的掩码——nbr_all非空位置1
输出：
    se_vec:
    J:
'''
def training(output_h, output_h_match, loss_all, sample_w, loss_w, learning_rate, 
             epochs, pre_epochs, ILL, e, k, sampled_nbr_num, save_suffix, dimension, dimension_g, c, 
             train_batchnum, test_batchnum,
             test, M0, e1, e2, nbr_all, mask_all):
    from include.Test import get_hits, get_hits_new
    '''
    tf.train.AdamOptimizer()函数是Adam优化算法：是一个寻找全局最优点的优化算法，引入了二次方梯度校正。
    '''
    #整体损失函数最小化
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_all)
    #Ws最小化
    train_step_w = tf.train.AdamOptimizer(learning_rate).minimize(loss_w, var_list=[sample_w])
    print('initializing...')
    #保存和恢复变量
    saver = tf.compat.v1.train.Saver()
    #用于初始化图中全局变量的Op
    init = tf.global_variables_initializer()
    #开启一个线程
    sess = tf.Session()
    sess.run(init)
    print('running...')
    J = []
    #ILL：将种子对构建成列表
    '''
    ILL
        type:numpy.ndarray
        shape:(4500, 2)
        value:[[22129 37240]
                [ 5946 16446]
                [ 2449 12949]
                ...
                [21029 36364]
                [21047 35687]
                [ 7535 18035]]
    '''
    ILL = np.array(ILL)

    #t：种子对的长度
    t = len(ILL)
    #ILL_reshape：将ILL转换为所需的尺寸
    ILL_reshape = np.reshape(ILL, 2 * t, order='F')
    '''
    np.ones()函数返回给定形状和数据类型的新数组，其中元素的值为1

    ILL[:, 0]
        type:       numpy.ndarray
        shape:      (4500,)
        value:      [22129  5946  2449 ... 21029 21047  7535]

    ILL[:, 0].reshape((t, 1))
        type:       numpy.ndarray
        shape:      (4500, 1)
        value:[[22129][ 5946][ 2449]...[21029][21047][ 7535]]

    L
        type:      numpy.ndarray 
        shape:     (4500, 125)
        value:      [[22129. 22129. 22129. ... 22129. 22129. 22129.]
                    [ 5946.  5946.  5946. ...  5946.  5946.  5946.]
                    [ 2449.  2449.  2449. ...  2449.  2449.  2449.]
                    ...
                    [21029. 21029. 21029. ... 21029. 21029. 21029.]
                    [21047. 21047. 21047. ... 21047. 21047. 21047.]
                    [ 7535.  7535.  7535. ...  7535.  7535.  7535.]]
    '''
    L = np.ones((t, k)) * (ILL[:, 0].reshape((t, 1)))
    '''
    neg_left
        type:   numpy.ndarray
        shape:  (562500,)
        value:  [22129. 22129. 22129. ...  7535.  7535.  7535.]
    '''
    #出现在种子实体对中左边的实体
    #t：ILL中的种子对数量；k：每一个种子对对应的负面消极对的数量
    neg_left = L.reshape((t * k,))

    '''
    ILL[:, 1]
        type:       numpy.ndarray
        shape:      (4500,)
        value:      [37240 16446 12949 ... 36364 35687 18035]

    ILL[:, 1].reshape((t, 1))
        type:       numpy.ndarray
        shape:      (4500, 1)
        value:[[37240][16446][12949]...[36364][35687][18035]]
    '''
    L = np.ones((t, k)) * (ILL[:, 1].reshape((t, 1)))
    # 出现在种子实体对中右边的实体
    neg2_right = L.reshape((t * k,))

    '''
    输入：
        M0：             表示基于索引的双向KG
        e：              两个知识库实体个数和
        sampled_nbr_num：邻居的数量
    '''
    '''
    输出：
        nbr_sampled：    每一个节点对应的邻居形成的矩阵
        mask_sampled：   nbr_sampled矩阵的掩码矩阵
    '''
    nbr_sampled, mask_sampled = get_nbr(M0, e, sampled_nbr_num)
    '''
    输入：
        e：两个知识库内实体的总数
        e1：源知识库
        e2：目标知识库
    '''
    '''
    输出：
        mask_e1：长度等于e，e1中的实体对应的索引位,置1
        mask_e2：长度等于e，e2中的实体对应的索引位,置1
    '''
    mask_e1, mask_e2 = mask_candidate(e, e1, e2)
    test_reshape = np.reshape(np.array(test), -1, order='F')
    #返回一个维度为dimension【GCN层的隐藏表征的维度：300】的对角阵（轴线为1，其余位为0）
    sample_w_vec = np.identity(dimension)
    test_can_num=50

    if not os.path.exists("model/"):
        os.makedirs("model/")
    '''
    if os.path.exists("model/save_"+save_suffix+".ckpt.meta"):
        saver.restore(sess, "model/save_"+save_suffix+".ckpt")
        start_epoch=pre_epochs
    '''
    if os.path.exists("model/save"):
        saver.restore(sess, "model/save_"+save_suffix+".ckpt")
        start_epoch=pre_epochs
    else:
        start_epoch=0

    #起始：0   终点：:50
    for i in range(start_epoch, epochs):
        if i % 50 == 0:
            '''
            函数：run(fetches,feed_dict=None,options=None,run_metadata=None)当构建完图后，需要在一个session会话中启动图，
            第一步是创建一个Session对象。为了取回（Fetch）操作的输出内容, 可以在使用 Session 对象的 run()调用执行图时，
            传入一些 tensor, 这些 tensor 会帮助你取回结果。在python语言中，返回的tensor是numpy ndarray对象
            '''
            '''
            output_h
                type:tensor
                shape:(38960, 300)
                value:Tensor("add_3:0", shape=(38960, 300), dtype=float32)
            '''
            '''
            out：以数组的形式表示GCN的输出层，38900等于两个知识库内所有的双向三元组头尾实体存储的数量
                    看不懂上面那句话，去看看M和M0
                type:numpy.ndarray
                shape:(38960, 300)
                value:[[-0.01299374 -0.01221544  0.01501506 ...  0.00236383  0.0320911  -0.00371881]
                        [-0.00379386  0.01567059 -0.00581796 ...  0.02368146  0.0388569  0.02662149]
                        ...
                        [-0.01046023 -0.01601668  0.06152889 ...  0.0034243   0.0307948  0.02681164]
                        [-0.00511091  0.00670491 -0.01826509 ... -0.00405984  0.00149903 0.00467891]]
            '''
            out = sess.run(output_h)
            print('get negative pairs')
            '''
            neg2_left
                type:numpy.ndarray'
                shape:(562500,)
                value:[37240 13124 22129 ... 20679  2978 18698]
            '''
            '''
            输入：
                ILL[:, 1]：右边的种子实体对
                out：以数组的形式表示GCN的输出层
                k：消极对的数量
                train_batchnum：训练集邻域采样数量 
            输出：
                从全部实体中筛选，根据曼哈顿距离，返回距离种子实体对右边的实体最远的实体
            '''
            neg2_left = get_neg(ILL[:, 1], out, k, train_batchnum)

            #从全部实体中筛选，根据曼哈顿距离，返回距离种子实体对左边的实体最远的实体
            neg_right = get_neg(ILL[:, 0], out, k, train_batchnum)

            print('sample candidates')
            '''
            输入:
                ILL[:, 1]:
                ILL[:, 0]:
                out:以数组的形式表示GCN的输出层
                c:每个实体的候选集个数
                mask_e2:长度等于e，e2中的实体对应的索引位,置1
                train_batchnum:训练集邻域采样数量
            输出:
                c_left:种子实体对右边的候选实体，其中的元素属于 种子实体对左边以及
                根据全部实体与种子实体对右边的实体 的曼哈顿距离随机抽取（抽取概率与相关度成正比）的实体
                c_right:种子实体对左边的候选实体，其中的元素属于 种子实体对右边以及
                根据全部实体与种子实体对左边的实体 的曼哈顿距离随机抽取（抽取概率与相关度成正比）的实体
            '''
            c_left = sample_candidate(ILL[:, 1], ILL[:, 0], out, c, mask_e2, train_batchnum)
            c_right = sample_candidate(ILL[:, 0], ILL[:, 1], out, c, mask_e1, train_batchnum)

            #将左右候选实体进行拼接，变成一个2*len(ILL)*c的三维矩阵
            #2代表左右；len(ILL)种子实体个数；c代表候选实体数目
            candidate = np.reshape(np.concatenate((c_right, c_left), axis=0), (2, len(ILL), c))
            print('sample neighborhood')
            '''
            输入：
                out：GCN输出，数量等于两个知识库内所有双向实体对的个数，(38960, 300)
                nbr_all：每一个节点对应的邻居的编码
                mask_all：nbr_all的掩码——nbr_all非空位置1
                e：两个知识库实体个数和
                sampled_nbr_num：邻居的数量
                sample_w_vec：返回一个维度为dimension【GCN层的隐藏表征的维度：300】的对角阵（轴线为1，其余位为0）
                test_batchnum：测试集邻域采样数量
            输出：
                nbr_sampled:根据注意力权重，从nbr_all中随机抽取邻居
                mask_sampled:nbr的掩码矩阵
            '''
            nbr_sampled, mask_sampled = sample_nbr(
                out, nbr_all, mask_all, e, sampled_nbr_num, sample_w_vec, test_batchnum)
            #定义一个字典，该字典内的数值针对的是全体内容
            #type:dict
            feeddict = {"ILL:0": ILL,
                        "candidate:0": candidate.reshape((-1,)),
                        #出现在种子实体对中左边的实体
                        "neg_left:0": neg_left,
                        #从全部实体中筛选，根据曼哈顿距离，返回距离种子实体对左边的实体最远的实体
                        "neg_right:0": neg_right,
                        #从全部实体中筛选，根据曼哈顿距离，返回距离种子实体对右边的实体最远的实体
                        "neg2_left:0": neg2_left,
                        # 出现在种子实体对中右边的实体
                        "neg2_right:0": neg2_right,
                        #nbr_sampled: 根据注意力权重，从nbr_all中随机抽取邻居
                        "nbr_sampled:0": nbr_sampled,
                        # mask_sampled: nbr的掩码矩阵
                        "mask_sampled:0": mask_sampled,
                        "c:0": c}
            #如果i在预训练批次中，字典中置0，否则置1
            if i < pre_epochs:
                feeddict["alpha:0"] = 0
            else:
                feeddict["alpha:0"] = 1

        #根据训练集邻域采样数量进行循环
        for j in range(train_batchnum):
            #t：种子对的长度
            beg = int(t / train_batchnum * j)
            if j==train_batchnum-1:
                end=t
            else:
                end = int(t / train_batchnum * (j + 1))
            #根据每一个batch，缩减feeddict的范围与内容多少
            feeddict["ILL:0"] = ILL[beg:end]
            feeddict["candidate:0"] = candidate[:, beg:end].reshape((-1,))
            feeddict["neg_left:0"] = neg_left.reshape(
                (t, k))[beg:end].reshape((-1,))
            feeddict["neg_right:0"] = neg_right.reshape(
                (t, k))[beg:end].reshape((-1,))
            feeddict["neg2_left:0"] = neg2_left.reshape(
                (t, k))[beg:end].reshape((-1,))
            feeddict["neg2_right:0"] = neg2_right.reshape(
                (t, k))[beg:end].reshape((-1,))
            _ = sess.run([train_step], feed_dict=feeddict)

        if i == pre_epochs - 1:
            #定义存储节点的路径
            save_path = saver.save(sess, "model/save_"+save_suffix+".ckpt")
            print("Save to path: ", save_path)

        if i % 10 == 0:
            print('%d/%d' % (i + 1, epochs), 'epochs...')
            outvec = sess.run(output_h, feed_dict=feeddict)
            '''
            输入：
               outvec：以当前batch筛选的feeddict作为参数，产生的GCN输出层的tensor
               test：70%的测试集
               test_can_num： 50
            输出：
                test_can：基于曼哈顿距离，得到的测试集中知识库一与知识库二在对方知识库中距离最近的实体
            '''
            test_can = get_hits(outvec, test, test_can_num)
            #预训练批次之后
            if i >= pre_epochs:
                #每一个训练批次内进行自迭代
                for j in range(test_batchnum):
                    beg = int(len(test) / test_batchnum * j)
                    if j==test_batchnum-1:
                        end=len(test)
                    else:
                        end = int(len(test) / test_batchnum * (j + 1))
                    feeddict_test = {"ILL:0": test[beg:end],
                                     "candidate:0": test_can[:, beg:end].reshape((-1,)),
                                     "nbr_sampled:0": nbr_sampled,
                                     "mask_sampled:0": mask_sampled,
                                     "c:0": test_can_num}
                    #参数为feeddict_test的，邻域聚集阶段的输出层
                    outvec_h_match = sess.run(
                        output_h_match, feed_dict=feeddict_test)
                    if j == 0:
                        #在预训练之后，第一次正式训练时，将outvec_h_match转换为所需要的尺寸
                        outvec_h_match_all = outvec_h_match.reshape((2, -1, dimension+dimension_g))
                    else:
                        #在非第一次训练时，将上一批次的outvec_h_match_all，与本批次的outvec_h_match进行结合
                        outvec_h_match_all = np.concatenate(
                            [outvec_h_match_all, outvec_h_match.reshape((2, -1, dimension+dimension_g))], axis=1)
                '''
                outvec_h_match_all:该test_batchnum内每一次循环生成的邻域聚集输出层
                test_can:基于曼哈顿距离，得到的测试集中知识库一与知识库二在对方知识库中距离最近的实体
                test:70%的测试集
                test_can_num:50
                '''
                get_hits_new(outvec_h_match_all, test_can, test, test_can_num)

        #预训练结束，并且还有一个批次就完成本batch的训练
        if i >= pre_epochs and i % 50 == 49:
            print('train sample w')
            for _ in range(10):
                #随机从种子对中选择十个拿出来
                select_train = np.random.choice(len(ILL), 10)
                feeddict["select_train:0"] = select_train
                for j in range(5):
                    global thw
                    _, thw = sess.run([train_step_w, loss_w],
                                      feed_dict=feeddict)
                print(thw)
            sample_w_vec = sess.run(sample_w, feed_dict=feeddict)

    sess.close()
    return outvec, J
