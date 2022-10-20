import tensorflow as tf
import argparse
from include.Config import Config
from include.Model import build, training, get_nbr
from include.Load import *

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

seed = 12306
np.random.seed(seed)
tf.set_random_seed(seed)

'''
Followed the code style of HGCN-JE-JR:
https://github.com/StephanieWyt/HGCN-JE-JR
'''

parser = argparse.ArgumentParser()
#parser.add_argument('--dataset', type=str, help='DBP15k or DWY100k')
#parser.add_argument('--lang', type=str, help='zh_en, ja_en and fr_en for DBP15K , dbp_wd and dbp_yg for DWY100K')
parser.add_argument('--dataset', type=str, help='DBP15k or DWY100k')
parser.add_argument('--lang', type=str, help='zh_en, ja_en and fr_en for DBP15K , dbp_wd and dbp_yg for DWY100K')
args = parser.parse_args()
#下面两行是我自己加的
args.dataset ='DBP15k'
args.lang = 'zh_en'
if __name__ == '__main__':
    config = Config(args.dataset,args.lang)
    #源知识图的实体，set函数创建一个object
    e1 = set(loadfile(config.e1, 1))
    #目标知识图的实体
    e2 = set(loadfile(config.e2, 1))
    #e1与e2实体个数求和
    #e1 = 19388 ; e2 = 19572 ; e = 38960
    e = len(e1 | e2)

    #加入种子实体:illL = 15000
    ILL = loadfile(config.ill, 2)
    illL = len(ILL)

    #shuffle直接在原来的数组上随机排序
    np.random.shuffle(ILL)
    '''
    train:
        type:<class 'numpy.ndarray'>
        len:4500
        value:[[22129 37240]
               [ 5946 16446]
               [ 2449 12949]
               ...
               [21029 36364]
               [21047 35687]
               [ 7535 18035]]
    test:
        type:<class 'list'>
        len:10500
        value:[(4946, 15446), (2132, 12632), (1233, 11733)……(23426, 37697), (8848, 19348)]        
    '''
    #30%的训练集，70%的训练集
    train = np.array(ILL[:illL // 10 * config.seed])
    test = ILL[illL // 10 * config.seed:]


    # load a file and return a list of tuple containing $num integers in each line
    '''
    注：这里的知识库是{h,r,t}的知识库，利用索引表示，对应于ent_ids_1&2与ref_ent_ids
    '''
    KG1 = loadfile(config.kg1, 3)
    KG2 = loadfile(config.kg2, 3)

    '''
    输入：
        config.dim：GCN层的隐藏表征的维度是300
        config.dim_g:邻域表征的维度是50
        config.act_func: GCN激活函数，对应文章中的等式一
        config.gamma:公式10中的γ
        config.k:消极对的数量
        config.vec: 训练好的词向量
        e：两个知识库实体个数和
        config.all_nbr_num：所有可以被涉及的邻居的数量 
        config.sampled_nbr_num：被采样的邻居的数量 
        config.beta：β，论文中公式（6）涉及的参数，加权匹配向量的参数。
        KG1 + KG2：源知识库与目标知识库的结合
    '''
    '''
    输出：
        output_h：GCN网络的输出层
        output_h_match：邻域聚集阶段的输出层
        loss_all ：整体的损失函数
        sample_w：聚合采样的所有的邻居信息
        loss_w：训练Ws的损失函数
        M0：表示基于索引的双向KG
        
        M = {(1, 4): 1, (4, 1): 1, (5, 8): 1, (8, 5): 1,(2, 7): 1, (7, 2): 1,(9, 2): 1, (2, 9): 1,(6, 3): 1, (3, 6): 1}
        nbr=[[], [4], [7, 9], [6], [1], [8], [3], [2], [5], [2]]
        nbr_all=[[4, 0, 0], [0, 0, 0], [7, 8, 0], [6, 0, 0], [0, 0, 0], [8, 0, 0], [3, 0, 0], [2, 0, 0], [5, 2, 0], [0, 0, 0]]
        mask_all=[[1, 0, 0], [0, 0, 0], [1, 1, 0], [1, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 0]]
        
        nbr_all：每一个节点对应的邻居的编码 
        mask_all：nbr_all的掩码
    '''
    output_h, output_h_match, loss_all, sample_w, loss_w, M0, nbr_all, mask_all = \
        build(config.dim, config.dim_g, config.act_func, config.gamma,
            config.k, config.vec, e, 
            config.all_nbr_num, config.sampled_nbr_num, config.beta, KG1 + KG2)
    '''
    se_vec, J = training(output_h, output_h_match, loss_all, sample_w, loss_w, config.lr, 
                         config.epochs, config.pre_epochs, train, e,
                         config.k, config.sampled_nbr_num, config.save_suffix, config.dim, config.dim_g, 
                         config.c, config.train_batchnum, config.test_batchnum, 
                         test, M0, e1, e2, nbr_all, mask_all)
    '''
    '''
    输入：
        output_h:GCN网络的输出层
        output_h_match:邻域聚集阶段的输出层
        loss_all:整体的损失函数
        sample_w:聚合采样的所有的邻居信息
        loss_w:训练Ws的损失函数
        config.lr:学习率
        config.epochs:训练批次
        config.pre_epochs:预训练（也就是GCN）的训练批次
        train:30%的训练集
        e:两个知识库实体个数和
        config.k:消极对的数量
        config.sampled_nbr_num:邻居的数量
        config.save_suffix:数据集与语言：DBP15k_zh_en
        config.dim:GCN层的隐藏表征的维度是300
        config.dim_g:邻域表征的维度是50
        config.c:每个实体的候选集个数
        config.train_batchnum:训练集邻域采样数量 
        config.test_batchnum:测试集邻域采样数量
        test:70%的测试集
        M0:表示基于索引的双向KG
        e1:源知识库
        e2:目标知识库
        nbr_all:每一个节点对应的邻居的编码
        mask_all:nbr_all的掩码——nbr_all非空位置1
    输出：
        se_vec:每当十的整数次时，以当前batch筛选的feeddict作为参数，产生的GCN输出层的tensor
        J:
    '''
    se_vec, J= training(output_h, output_h_match, loss_all, sample_w, loss_w, config.lr,
                         config.epochs, config.pre_epochs, train, e,
                         config.k, config.sampled_nbr_num, config.save_suffix, config.dim, config.dim_g,
                         config.c, config.train_batchnum, config.test_batchnum,
                         test, M0, e1, e2, nbr_all, mask_all)
    print("se_vec")
    print(se_vec)

