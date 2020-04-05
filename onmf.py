# (setq python-shell-interpreter "./venv/bin/python")


# import tensorflow as tf
import numpy as np
import progressbar
# import imageio
import matplotlib.pyplot as plt
from numpy import linalg as LA
from time import time

from sklearn.decomposition import SparseCoder





DEBUG = False


class Online_NMF():

    def __init__(self,
                 X,
                 n_components=100,
                 iterations=500,
                 batch_size=20,
                 ini_dict=None,
                 ini_A=None,
                 ini_B=None,
                 ini_C=None,
                 history=0,
                 alpha=None,
                 beta=None,
                 subsample=False):
        '''
        X: data matrix
        n_components (int): number of columns in dictionary matrix W where each column represents on topic/feature
        iter (int): number of iterations where each iteration is a call to step(...)
        batch_size (int): number random of columns of X that will be sampled during each iteration
        '''
        self.X = X
        self.n_components = n_components
        self.batch_size = batch_size
        self.iterations = iterations
        self.subsample = subsample
        self.initial_dict = ini_dict
        self.initial_A = ini_A
        self.initial_B = ini_B
        self.initial_C = ini_C
        self.history = history
        self.alpha = alpha
        self.beta = beta
        self.code = np.zeros(shape=(n_components, X.shape[1]))

    def sparse_code(self, X, W):
        '''
        Given data matrix X and dictionary matrix W, find 
        code matrix H such that W*H approximates X

        args:
            X (numpy array): data matrix with dimensions: features (d) x samples (n)
            W (numpy array): dictionary matrix with dimensions: features (d) x topics (r)

        returns:
            H (numpy array): code matrix with dimensions: topics (r) x samples(n)
        '''

        if DEBUG:
            print('sparse_code')
            print('X.shape:', X.shape)
            print('W.shape:', W.shape, '\n')

        # initialize the SparseCoder with W as its dictionary
        # then find H such that X \approx W*H
        if self.alpha == None:
            coder = SparseCoder(dictionary=W.T, transform_n_nonzero_coefs=None,
                                transform_alpha=2, transform_algorithm='lasso_lars', positive_code=True)
        else:
            coder = SparseCoder(dictionary=W.T, transform_n_nonzero_coefs=None,
                                transform_alpha=self.alpha, transform_algorithm='lasso_lars', positive_code=True)
        # alpha = L1 regularization parameter.
        H = coder.transform(X.T)

        # transpose H before returning to undo the preceding transpose on X
        return H.T

    def update_dict(self, W, A, B):
        '''
        Updates dictionary matrix W using new aggregate matrices A and B

        args:
            W (numpy array): dictionary matrix with dimensions: data_dim (d) x topics (r)
            A (numpy array): aggregate matrix with dimensions: topics (r) x topics(r)
            B (numpy array): aggregate matrix with dimensions: topics (r) x data_dim (d)

        returns:
            W1 (numpy array): updated dictionary matrix with dimensions: features (d) x topics (r)
        '''
        # extract matrix dimensions from W
        # and initializes the copy W1 that is updated in subsequent for loop
        d, r = np.shape(W)
        W1 = W.copy()

        #****
        for j in np.arange(r):
                # W1[:,j] = W1[:,j] - (1/W1[j,j])*(np.dot(W1, A[:,j]) - B.T[:,j])
                W1[:,j] = W1[:,j] - (1/(A[j,j]+1) )*(np.dot(W1, A[:,j]) - B.T[:,j])
                W1[:,j] = np.maximum(W1[:,j], np.zeros(shape=(d, )))
                W1[:,j] = (1/np.maximum(1, LA.norm(W1[:,j])))*W1[:,j]
        
        return W1


    def step(self, X, A, B, C, W, t):
        '''
        Performs a single iteration of the online NMF algorithm from
        Han's Markov paper. 
        Note: H (numpy array): code matrix with dimensions: topics (r) x samples(n)

        args:
            X (numpy array): data matrix with dimensions: data_dim (d) x samples (n)
            A (numpy array): aggregate matrix with dimensions: topics (r) x topics(r)
            B (numpy array): aggregate matrix with dimensions: topics (r) x data_dim (d)
            W (numpy array): dictionary matrix with dimensions: data_dim (d) x topics (r)
            t (int): current iteration of the online algorithm
        
        returns:
            Updated versions of H, A, B, and W after one iteration of the online NMF
            algorithm (H1, A1, B1, and W1 respectively)
        '''
        d, n = np.shape(X)
        d, r = np.shape(W)
        
        # Compute H1 by sparse coding X using dictionary W
        H1 = self.sparse_code(X, W)

        if DEBUG:
            print(H1.shape)

        # Update aggregate matrices A and B
        t = t.astype(float)
        if self.beta == None:
            beta = 1
        else:
            beta = self.beta
        A1 = (1 - (t ** (-beta))) * A + t ** (-beta) * np.dot(H1, H1.T)
        B1 = (1 - (t ** (-beta))) * B + t ** (-beta) * np.dot(H1, X.T)
        C1 = (1 - (t ** (-beta))) * C + t ** (-beta) * np.dot(X, X.T)

        # Update dictionary matrix
        W1 = self.update_dict(W, A, B)
        self.history = t+1
        # print('history=', self.history)
        return H1, A1, B1, C1, W1

    def train_dict(self):
        '''
        Learns a dictionary matrix W with n_components number of columns based 
        on a fixed data matrix X
        
        args:
            X (numpy array): data matrix with dimensions: data_dim (d) x samples (n)


        return:
            W (numpy array): dictionary matrix with dimensions: data_dim (d) x topics (r)
        '''
        # extract matrix dimensions from X
        d, n = np.shape(self.X)
        r = self.n_components
        code = self.code

        if self.initial_dict is None:
            # initialize dictionary matrix W with random values
            # and initialize aggregate matrices A, B with zeros
            W = np.random.rand(d, r)
            A = np.zeros((r, r))
            B = np.zeros((r, d))
            C = np.zeros((d, d))
            t0 = self.history
        else:
            W = self.initial_dict
            A = self.initial_A
            B = self.initial_B
            C = self.initial_C
            t0 = self.history

        for i in np.arange(1, self.iterations):
            idx = np.arange(self.X.shape[1])
            # randomly choose batch_size number of columns to sample
            # initializing the "batch" of X, which are the subset
            # of columns from X that were randomly chosen above
            if self.subsample:
                idx = np.random.randint(n, size=self.batch_size)

            X_batch = self.X[:, idx]
            # iteratively update W using batches of X, along with
            # iteratively updated values of A and B
            H, A, B, C, W = self.step(X_batch, A, B, C, W, t0+i)
            code[:, idx] += H
            # print('dictionary=', W)
            # print('code=', H)
            # plt.matshow(H)
        print('iteration %i out of %i' % (i, self.iterations))
        return W, A, B, C, code



### Used only for an old version of Ising model simulation
class Online_NMF_stack():
    # ONMF for a stack of data matrices representing a time series of data matrices
    def __init__(self, X, n_components=100, iterations=500, batch_size=20, ini_dict=None):
        '''
        X: time series of data matrices (3d array)
        n_components (int): number of columns in dictionary matrix W where each column represents on topic/feature
        iter (int): number of iterations where each iteration is a call to step(...) for each slice of the stack
        batch_size (int): number random of columns of X that will be sampled during each iteration  per slice
        '''
        self.X = X
        self.n_components = n_components
        self.batch_size = batch_size
        self.iterations = iterations
        self.initial_dict = ini_dict

    def sparse_code(self, X, W):
        '''
        Given data matrix X and dictionary matrix W, find
        code matrix H such that W*H approximates X

        args:
            X (numpy array): data matrix with dimensions: data_dim (d) x samples (n)
            W (numpy array): dictionary matrix with dimensions: data_dim (d) x topics (r)

        returns:
            H (numpy array): code matrix with dimensions: topics (r) x samples(n)
        '''

        if DEBUG:
            print('sparse_code')
            print('X.shape:', X.shape)
            print('W.shape:', W.shape, '\n')

        # extract matrix dimensions from X, W
        # and initialize H with appropriate dimensions
        d, n = np.shape(X)
        d, r = np.shape(W)
        H = np.zeros([n, r])

        # initialize the SparseCoder with W as its dictionary
        # then find H such that X \approx W*H
        coder = SparseCoder(dictionary=W.T, transform_n_nonzero_coefs=None,
                            transform_alpha=2, transform_algorithm='lasso_lars', positive_code=False)
        H = coder.transform(X.T)
        # transpose H before returning to undo the preceding transpose on X
        return H.T

    def update_dict(self, W, A, B):
        '''
        Updates dictionary matrix W using new aggregate matrices A and B

        args:
            W (numpy array): dictionary matrix with dimensions: data_dim (d) x topics (r)
            A (numpy array): aggregate matrix with dimensions: topics (r) x topics(r)
            B (numpy array): aggregate matrix with dimensions: topics (r) x data_dim(d)

        returns:
            W1 (numpy array): updated dictionary matrix with dimensions: data_dim (d) x topics (r)
        '''
        # extract matrix dimensions from W
        # and initializes the copy W1 that is updated in subsequent for loop
        d, r = np.shape(W)
        W1 = W.copy()

        # ****
        for j in np.arange(r):
            # W1[:,j] = W1[:,j] - (1/W1[j,j])*(np.dot(W1, A[:,j]) - B.T[:,j])
            W1[:, j] = W1[:, j] - (1 / (A[j, j] + 1)) * (np.dot(W1, A[:, j]) - B.T[:, j])
            W1[:, j] = np.maximum(W1[:, j], np.zeros(shape=(d,)))
            W1[:, j] = (1 / np.maximum(1, LA.norm(W1[:, j]))) * W1[:, j]

        return W1

    def step(self, X, A, B, W, t):
        '''
        Performs a single iteration of the online NMF algorithm
        Note: H (numpy array): code matrix with dimensions: topics (r) x samples(n)

        args:
            X (numpy array): data matrix with dimensions: data_dim (d) x samples (n)
            A (numpy array): aggregate matrix with dimensions: topics (r) x topics(r)
            B (numpy array): aggregate matrix with dimensions: topics (r) x data_dim(d)
            W (numpy array): dictionary matrix with dimensions: data_dim (d) x topics (r)
            t (int): current iteration of the online algorithm

        returns:
            Updated versions of H, A, B, and W after one iteration of the online NMF
            algorithm (H1, A1, B1, and W1 respectively)
        '''
        d, n = np.shape(X)
        d, r = np.shape(W)

        # Compute H1 by sparse coding X using dictionary W
        H1 = self.sparse_code(X, W)

        if DEBUG:
            print(H1.shape)

        # Update aggregate matrices A and B
        A1 = (1 / t) * ((t - 1) * A + np.dot(H1, H1.T))
        B1 = (1 / t) * ((t - 1) * B + np.dot(H1, X.T))
        # Update dictionary matrix
        W1 = self.update_dict(W, A, B)

        return H1, A1, B1, W1

    def train_dict(self):
        '''
        Learns a dictionary matrix W with n_components number of columns based
        on a fixed stack of data matrices X from bottom to top

        args:
            X (numpy array): data matrix with dimensions: data_dim (d) x samples (n) x num_stack (m)

        return:
            W (numpy array): dictionary matrix with dimensions: data_dim (d) x topics (r)
        '''
        # extract matrix dimensions from X
        d, n, m = np.shape(self.X)
        r = self.n_components

        if self.initial_dict is None:
            # initialize dictionary matrix W with random values
            # and initialize aggregate matrices A, B with zeros
            W = np.random.rand(d, r)
            A = np.zeros((r, r))
            B = np.zeros((r, d))
            history = 0
        else:
            W = self.initial_dict
            A = np.zeros((r, r))
            B = np.zeros((r, d))

        for j in np.arange(1, m):
            for i in np.arange(1, self.iterations):
                # randomly choose batch_size number of columns to sample
                idx = np.random.randint(n, size=self.batch_size)

                # initializing the "batch" of X, which are the subset
                # of columns from X that were randomly chosen above
                X_batch = self.X[:, idx, j]

                # iteratively update W using batches of X, along with
                # iteratively updated values of A and B
                H, A, B, W = self.step(X_batch, A, B, W, i+(j*self.iterations))
        return W, A, B
