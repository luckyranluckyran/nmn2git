import numpy as np
import scipy.spatial
import pickle as pkl

'''
输入：
   vec：outvec：以当前batch筛选的feeddict作为参数，产生的GCN输出层的tensor
   test_pair：test：70%的测试集
   c：test_can_num： 50
输出：
    test_can：基于曼哈顿距离，得到的测试集中知识库一与知识库二在对方知识库中距离最近的实体
'''
def get_hits(vec, test_pair, c, top_k=(1, 10)):
    #L:测试集中的知识库一中的实体
    #R:测试集中的知识库二中的实体
    L = np.array([e1 for e1, e2 in test_pair])
    R = np.array([e2 for e1, e2 in test_pair])
    '''
    Lvec:测试集中的知识库一中的实体对应的GCN输出
    Rvec:测试集中的知识库二中的实体对应的GCN输出
    '''
    Lvec = vec[L]
    Rvec = vec[R]
    #测试集中的知识库一与知识库二中实体的曼哈顿距离
    sim = scipy.spatial.distance.cdist(Lvec, Rvec, metric='cityblock')
    top_lr = [0] * len(top_k)
    candidate = []
    for i in range(Lvec.shape[0]):
        #将矩阵a按照axis排序，并返回排序后的下标
        rank = sim[i, :].argsort()
        rank_index = np.where(rank == i)[0][0]
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1
        #将测试集中的知识库二中的实体的前c个值纳入候选实体中
        candidate.append(R[rank[0:c]])

    top_rl = [0] * len(top_k)
    for i in range(Rvec.shape[0]):
        rank = sim[:, i].argsort()
        rank_index = np.where(rank == i)[0][0]
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_rl[j] += 1
        candidate.append(L[rank[0:c]])

    print('For each left (KG structure embedding):')
    for i in range(len(top_lr)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100))

    return np.array(candidate).reshape((2, -1, c))


'''
outvec_h_match_all:该test_batchnum内每一次循环生成的邻域聚集输出层
test_can:基于曼哈顿距离，得到的测试集中知识库一与知识库二在对方知识库中距离最近的实体
test:70%的测试集
test_can_num:50
'''
def get_hits_new(vec, candidate, test_pair, c, top_k=(1, 10)):
    t = len(test_pair)
    L = np.array([e1 for e1, e2 in test_pair])
    #R测试集中属于知识库二的实体
    R = np.array([e2 for e1, e2 in test_pair])

    vec = np.reshape(vec, (2, t, 2, c, -1))
    Lvec = vec[0, :, 0]
    Rvec = vec[1, :, 0]
    sim = np.sum(np.abs(Lvec - Rvec), -1)
    candidate_L = np.reshape(candidate, (2, t, -1))[0]

    top_lr = [0] * len(top_k)
    for i in range(t):
        x = -1
        for j in range(len(candidate_L[i])):
            if R[i] == candidate_L[i][j]:
                x = j
        if x >= 0:
            rank = sim[i].argsort()
            rank_index = np.where(rank == x)[0][0]
            for j in range(len(top_k)):
                if rank_index < top_k[j]:
                    top_lr[j] += 1

    print('For each left (NMN):')
    for i in range(len(top_lr)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100))
