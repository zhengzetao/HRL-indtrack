import numpy as np
import pandas as pd
from cvxopt import matrix
from cvxopt import solvers

# Non verbose
solvers.options['show_progress'] = False

class qp_solver:
    def __init__(self, df:pd.DataFrame, limits:np.ndarray=None, col_index:str='index'):
        self.df = df.copy()
        self.col_index = col_index
        self.weights = df.loc[:, ~df.columns.str.match(self.col_index)].columns.to_numpy().tolist()
        self.limits = limits
    
    def _H_matrix(self):
        df = self.df.copy()
        df = df.loc[:, ~df.columns.str.match(self.col_index)]
        N = df.shape[1]
        T = df.shape[0]
        colnames = df.columns.to_numpy()
        
        H_mat = np.zeros((N, N))
        for i, col_i in enumerate(colnames):
            for j, col_j in enumerate(colnames):
                value = np.dot(df[col_i].copy().to_numpy() ,
                               df[col_j].copy().to_numpy()) / T
                H_mat[i, j] = value
        return H_mat
    
    def _g_matrix(self):
        df = self.df.copy()
        N = df.loc[:, ~df.columns.str.match(self.col_index)].shape[1]
        T = df.shape[0]
        
        colnames_not_index = df.loc[:, ~df.columns.str.match(self.col_index)].columns.to_numpy()
        
        g_vec = np.zeros(N)
        for i, col_i in enumerate(colnames_not_index):
            value = np.dot(df[col_i].copy().to_numpy(),
                           df[self.col_index].copy().to_numpy()) / T
            g_vec[i] = value
        return -g_vec
    
    def _linear_restrictions(self):
        df = self.df.copy()
        N = df.loc[:, ~df.columns.str.match(self.col_index)].shape[1]
        A = np.repeat(1, N)
        b = np.array([1])
        
        A = np.reshape(A, (1, N))
        b = np.reshape(b, (1,1))
        return A,b
    
    def _linear_inequealities(self):
        df = self.df.copy()
        N = df.loc[:, ~df.columns.str.match(self.col_index)].shape[1]
        Z = -np.identity(N)
        p = np.repeat([0], N).transpose()
        p = np.reshape(p, (N,1))
        
        return Z,p
    
    def solve(self):
        df = self.df.copy()
        N = df.loc[:, ~df.columns.str.match(self.col_index)].shape[1]
        
        H = matrix(self._H_matrix(), tc='d')
        g = matrix(self._g_matrix(), tc='d')
        
        A, b = self._linear_restrictions()
        A = matrix(A, tc='d')
        b = matrix(b, tc='d')
        
        Z, p = self._linear_inequealities()
        Z = matrix(Z, tc='d')
        p = matrix(p, tc='d')
        
        sol = solvers.qp(P=H,q=g, # objective
                         G=Z,h=p, # linear inequalities
                         A=A,b=b) # linear restrictions
        
        sol['x'] = np.array(sol['x'], dtype=float)
        
        # Adding objective cost value
        stock_data = df.loc[:, ~df.columns.str.match(self.col_index)].to_numpy()
        weights = sol['x']
        model_results = np.matmul(stock_data, weights).flatten()
        cost_value = np.mean((df[self.col_index].to_numpy() - model_results)**2)
        
        sol['cost value'] = cost_value
        
        return sol