# (setq python-shell-interpreter "./venv/bin/python")


# import tensorflow as tf
import numpy as np
import progressbar
# import imageio
import matplotlib.pyplot as plt
from numpy import linalg as LA
import tensorly as tl
from tensorly.tenalg import khatri_rao
from tensorly import unfold as tl_unfold
from tensorly.decomposition import parafac
from sklearn.decomposition import SparseCoder


DEBUG = False


class Online_NTF():

    def __init__(self,
                 X, n_components=100,
                 iterations=500,
                 sub_iterations = 10,
                 batch_size=20,
                 ini_dict=None,
                 ini_A=None,
                 ini_B=None,
                 history=0,
                 mode=0,
                 learn_joint_dict=False,
                 alpha=None):
        '''
        X: data tensor (3-dimensional)
        Seeks to find nonnegative tensor factorization X \approx \sum W^l * H^l * T^L
        W = (d by r), H = (n by r), T = (m by r)
        n_components (int) = r = number of columns in dictionary matrix W where each column represents on topic/feature
        iter (int): number of iterations where each iteration is a call to step(...)
        batch_size (int): number random of columns of X that will be sampled during each iteration
        '''
        self.X = X
        self.n_components = n_components
        self.batch_size = batch_size
        self.iterations = iterations
        self.sub_iterations = sub_iterations
        self.initial_dict = ini_dict
        self.initial_A = ini_A
        self.initial_B = ini_B
        self.history = history
        self.alpha = alpha
        self.mode = mode  # mode of unfolding the input tensor to learn marginal dictioanry by OMF problem
        self.learn_joint_dict = learn_joint_dict
        self.code = np.zeros(shape=(X.shape[1],n_components))

    def joint_sparse_code_tensor(self, X, W):
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
        return H

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

    def step(self, X, A, B, W, t):
        '''
        Performs a single iteration of the online NMF algorithm from
        Han's Markov paper. 
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

        # Compute H1 by sparse coding X using dictionary W
        H1 = self.joint_sparse_code_tensor(X, W)

        if DEBUG:
            print(H1.shape)

        # Update aggregate matrices A and B
        A1 = (1 / t) * ((t - 1) * A + np.dot(H1.T, H1))
        B1 = (1 / t) * ((t - 1) * B + np.dot(H1.T, X.T))
        # Update dictionary matrix
        W1 = self.update_dict(W, A, B)
        self.history = t + 1
        # print('history=', self.history)
        return H1, A1, B1, W1

    def train_dict_single(self):
        '''
        Given data tensor X and mode=i, learn dictionary matrix W and the complementary joint sparse code H.
        Reduce to matrix factorization by unfolding X along mode i

        ---------------if 'learn_joint_dict' = False:
        args:
            X (numpy array): data tensor with dimensions = (d) x (n) x (m)
            W (numpy array): dictionary matrix with dimensions  =  features (d) x topics (r) if mode=0
                                                                =  features (n) x topics (r) if mode=1
                                                                =  features (m) x topics (r) if mode=2

        method:
            X(i) = mode-i (Katri-Rao) unfolding of tensor X with dimensions = (d) x (n m) if mode = 0
                                                                            = (n) x (d m) if mode = 1
                                                                            = (m) x (d n) if mode = 2
            find sparse code H such that X(i) \approx W @ H using online matrix factorization

        returns:
            H (numpy array): code matrix with dimensions    = (r) x (n m) if mode = 0
                                                            = (r) x (d m) if mode = 1
                                                            = (r) x (d n) if mode = 2

                                                            if 'learn_joint_dict = False':

        ---------------if 'learn_joint_dict' = True:
        args:
            X (numpy array): data tensor with dimensions = (d) x (n) x (m)
            W (numpy array): dictionary matrix with dimensions  =  features (n m) x topics (r) if mode=0
                                                                =  features (d m) x topics (r) if mode=1
                                                                =  features (d n) x topics (r) if mode=2

        method:
            X(i) = mode-i (Katri-Rao) unfolding of tensor X with dimensions = (n m) x (d) if mode = 0
                                                                            = (d m) x (n) if mode = 1
                                                                            = (d n) x (m) if mode = 2
            find sparse code H such that X(i) \approx W @ H using online matrix factorization

        returns:
            H (numpy array): code matrix with dimensions    = (r) x (d) if mode = 0
                                                            = (r) x (n) if mode = 1
                                                            = (r) x (m) if mode = 2
        '''

        r = self.n_components
        code = self.code

        if not self.learn_joint_dict:
            X_unfold = tl_unfold(self.X, mode=self.mode)
            d, n = X_unfold.shape
        else:
            X_unfold = tl_unfold(self.X, mode=self.mode).T
            d, n = X_unfold.shape

        if self.initial_dict is None:
            # initialize dictionary matrix W with random values
            # and initialize aggregate matrices A, B with zeros
            W = np.random.rand(d, r)
            print('W.shape', W.shape)
            A = np.zeros((r, r))
            B = np.zeros((r, d))
            t0 = self.history
        else:
            W = self.initial_dict
            A = self.initial_A
            B = self.initial_B
            t0 = self.history

        for i in np.arange(1, self.iterations):
            # randomly choose batch_size number of columns to sample

            # initializing the "batch" of X, which are the subset
            # of columns from X_unfold that were randomly chosen above
            idx = np.random.randint(n, size=self.batch_size)
            X_batch = X_unfold[:, idx]
            # iteratively update W using batches of X, along with
            # iteratively updated values of A and B
            # print('X.shape before training step', self.X.shape)
            H, A, B, W = self.step(X_batch, A, B, W, t0+i)
            # code[idx,:] += H
            # print('dictionary=', W)
            # print('code=', H)
            # plt.matshow(H)

            #  progress status
            # print('Current iteration %i out of %i' % (i, self.iterations))
        return W, A, B, code
