from abc import ABC
import numpy as np
import pandas as pd


class Clusters(ABC):

    def __init__(self, stock_num, strategy='random', number=10):
        super(Clusters, self).__init__()
        self.strategy = strategy
        self.cluster_num = number
        self.cluster_lists = []
        self.stock_num = stock_num

    @property
    def cluster_statistic(self):
        stock_lists = range(self.stock_num)
        self.cluster_lists = [[] for i in range(self.cluster_num)]
        for i in range(self.stock_num // self.cluster_num):
            for j in range(self.cluster_num):
                self.cluster_lists[j].append(stock_lists[i*self.cluster_num+j])
        # for the remain items
        for i in range(self.stock_num // self.cluster_num * self.cluster_num, self.stock_num): self.cluster_lists[i%self.cluster_num].append(stock_lists[i])
        cluster_statistic = [len(self.cluster_lists[i]) for i in range(self.cluster_num)]
        
        return cluster_statistic

    def make_cluster(self, state):
        # if self.strategy == 'random':
        #     stock_num = state.shape[2]-1 # one is the index
        #     stock_lists = range(stock_num)
        #     self.cluster_lists = [[] for i in range(self.cluster_num)]
        #     for i in range(stock_num // self.cluster_num):
        #         for j in range(self.cluster_num):
        #             self.cluster_lists[j].append(stock_lists[i*self.cluster_num+j])
        #     # for the remain items
        #     for i in range(stock_num // self.cluster_num * self.cluster_num, stock_num): self.cluster_lists[i%self.cluster_num].append(stock_lists[i])
            
        if self.strategy == 'normal':
            # PERSON MATRIX : https://blog.csdn.net/qq_36810544/article/details/81363176
            # PERSON MATRIX : https://blog.csdn.net/sinat_35907936/article/details/123805702
            state = pd.DataFrame(state[0,:,:])
            state = state.pct_change().dropna().values
            index_value = state[:,0]
            stock_value = state[:,1:]

            # calculate the index-stocks correlation coefficients, a N*1 matrix
            
            def get_person(X, y):
                '''
                    input
                      vec 1: [dim, stock_num]
                      vec 2: [dim]
                    output 
                      vec: [stock_num,1]
                '''
                X = np.array(X)
                y = np.array(y)
                N = X.shape[1]
                X_center = X - X.mean(axis=0)
                X_std = X.std(axis=0)
                y_center = y - y.mean()
                y_std = y.std()
                _coef = np.dot(y_center, X_center) / (N * X_std * y_std)

                return _coef
            
            index_stock_coef = get_person(stock_value, index_value)
            N = index_stock_coef.shape[0]
            index_stock_coef = index_stock_coef.reshape(N,1)
            index_stock_coef_repeat = np.repeat(index_stock_coef, N, 1)

            # calculate the stocks-stocks correlation coefficients, a N*N matrix

            df = pd.DataFrame(data=stock_value)
            stock_stock_cor = df.corr(method='pearson')

            # calculate the coefficient
            coef = np.sum((stock_stock_cor.values[0,:] + index_stock_coef_repeat),axis=1) / N

            # from sklearn.preprocessing import MinMaxScaler
            # scaler = MinMaxScaler()
            # coef = scaler.fit_transform(coef)

            stock_score = []
            for stock_id, score in enumerate(index_stock_coef): stock_score.append((stock_id, score))
            sort_stock_score = sorted(stock_score, key=lambda x : x[1])
            sort_stock_list = [x[0] for x in sort_stock_score]

            # place the stock into pit

            # stock_lists = range(self.stock_num)
            self.cluster_lists = [[] for i in range(self.cluster_num)]

            for i in range(self.stock_num // self.cluster_num):
                for j in range(self.cluster_num):
                    self.cluster_lists[j].append(sort_stock_list[i*self.cluster_num+j])
            # for the remain items
            for i in range(self.stock_num // self.cluster_num * self.cluster_num, self.stock_num): self.cluster_lists[i%self.cluster_num].append(sort_stock_list[i])

        return self.cluster_lists