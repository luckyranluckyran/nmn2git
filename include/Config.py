import tensorflow as tf


class Config():
    def __init__(self, d='', l=''):
        dataset = d
        language = l
        if dataset=='DBP15k':
            prefix = 'data/DBP15k/' + language 
            self.kg1 = prefix + '/triples_1_s'
            self.kg2 = prefix + '/triples_2_s'
        else:
            prefix = 'data/' + str(dataset) + '/' + str(language)
            self.kg1 = prefix + '/triples_1'
            self.kg2 = prefix + '/triples_2'
        self.e1 = prefix + '/ent_ids_1'
        self.e2 = prefix + '/ent_ids_2'
        self.ill = prefix + '/ref_ent_ids'
        self.vec = prefix + '/vectorList.json'
        self.save_suffix = str(dataset)+'_'+str(language)
        print(str(dataset)+'_'+str(language))
        
        if dataset=='DWY100k':
            #self.epochs=200
            self.epochs=50
            self.pre_epochs = 50  # epochs to train the preliminary GCN
            self.train_batchnum=10
            self.test_batchnum=50
            self.all_nbr_num=20
        else:
            #self.epochs = 600
            self.epochs = 50
            self.pre_epochs = 500
            self.train_batchnum=1
            self.test_batchnum=5
            self.all_nbr_num=100

        #GCN层的隐藏表征的维度是300
        self.dim = 300
        #邻域表征的维度是50
        self.dim_g = 50
        #GCN激活函数，对应文章中的等式一
        self.act_func = tf.nn.relu
        #公式10中的γ
        self.gamma = 1.0  # margin based loss
        self.k = 125  # number of negative samples for each positive one
        self.seed = 3  # 30% of seeds
        #每个实体的候选集个数
        self.c = 20  # size of the candidate set
        #学习率
        self.lr = 0.001

        if dataset=='DBP15k':
            if language=='fr_en':
                #sampled_nbr_num表示邻居的数量
                self.sampled_nbr_num = 10  # number of sampled neighbors
            else:
                self.sampled_nbr_num = 3
            self.beta = 1  # weight of the matching vector
        else:
            self.sampled_nbr_num = 5
            #β=0.1
            self.beta = 0.1

