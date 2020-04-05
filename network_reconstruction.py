from onmf import Online_NMF
import numpy as np
import csv
import progressbar
import itertools
from time import time
from sklearn.decomposition import SparseCoder
import matplotlib.pyplot as plt

DEBUG = False


class Network_Reconstructor():
    def __init__(self,
                 source,
                 n_components=100,
                 MCMC_iterations=500,
                 sub_iterations=100,
                 loc_avg_depth=1,
                 sample_size=1000,
                 batch_size=10,
                 k1=1,
                 k2=2,
                 ntwk_size=211,
                 patches_file='',
                 alpha=None,
                 beta=None,
                 ONMF_subsample=False,
                 file_number=1):
        '''
        batch_size = number of patches used for training dictionaries per ONMF iteration
        sources: array of filenames to make patches out of
        patches_array_filename: numpy array file which contains already read-in images
        '''
        self.source = source
        self.n_components = n_components
        self.MCMC_iterations = MCMC_iterations
        self.sub_iterations = sub_iterations
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.loc_avg_depth = loc_avg_depth
        self.k1 = k1
        self.k2 = k2
        self.ntwk_size = ntwk_size
        self.patches_file = patches_file
        self.W = np.zeros(shape=((k1 + k2 + 1)**2, n_components))
        self.code = np.zeros(shape=(n_components, batch_size))
        self.alpha = alpha
        self.beta = beta
        self.ONMF_subsample = ONMF_subsample
        self.file_number = file_number

        # read in networks
        A = self.read_networks(source)
        self.A = A

    def read_networks(self, path):
        A = np.genfromtxt(path, usecols=range(self.ntwk_size))
        print('A_norm_full', np.linalg.norm(A/np.max(A)))
        ### get rid of singleton nodes
        for i in np.arange(self.ntwk_size):
            if A[i,i] == np.sum(A[i,:]) + np.sum(A[:,i]):
                A[i,i] = 0
                print('node is singleton:', i)
        A = A / np.max(A)
        print('A_norm', np.linalg.norm(A))
        return A

    def path_adj(self, k1, k2):
        # generates adjacency matrix for the path motif of k1 left nodes and k2 right nodes
        if k1 == 0 or k2 == 0:
            k3 = max(k1,k2)
            A = np.eye(k3 + 1, k=1, dtype=int)
        else:
            A = np.eye(k1+k2+1, k=1, dtype = int)
            A[k1,k1+1] = 0
            A[0,k1+1] = 1
        return A

    def indices(self, a, func):
        return [i for (i, val) in enumerate(a) if func(val)]

    def find_parent(self, B, i):
        # B = adjacency matrix of the tree motif rooted at first node
        # Nodes in tree B is ordered according to the depth-first-ordering
        # Find the index of the unique parent of i in B
        j = self.indices(B[:, i], lambda x: x == 1)  # indices of all neighbors of i in B
        # (!!! Also finds self-loop)
        return min(j)

    def tree_sample(self, B, x):
        # A = N by N matrix giving edge weights on networks
        # B = adjacency matrix of the tree motif rooted at first node
        # Nodes in tree B is ordered according to the depth-first-ordering
        # samples a tree B from a given pivot x as the first node

        A = self.A
        [N, N] = np.shape(A)
        [k, k] = np.shape(B)
        emb = np.array([x])  # initialize path embedding

        if sum(sum(B)) == 0:  # B is a set of isolated nodes
            y = np.random.randint(N, size=(1, k-1))
            y = y[0]  # juts to make it an array
            emb = np.hstack((emb, y))
        else:
            for i in np.arange(1, k):
                j = self.find_parent(B, i)
                if emb[j] == self.ntwk_size:
                    print(emb)

                if sum(A[emb[j], :]) > 0:
                    dist = A[emb[j], :] / sum(A[emb[j], :])
                    y = np.random.choice(np.arange(0, N), p=dist)
                else:
                    y = emb[j]
                    print('tree_sample_failed:isolated')
                emb = np.hstack((emb, y))

        return emb

    def grid_adj(self, rows, cols):
        n = rows * cols
        mx = np.zeros((n, n), dtype=int)
        for r in range(rows):
            for c in range(cols):
                i = r * cols + c
                # Two inner diagonals
                if c > 0:
                    mx[i - 1, i] = mx[i, i - 1] = 1
                # Two outer diagonals
                if r > 0:
                    mx[i - cols, i] = mx[i, i - cols] = 1
        return mx

    def torus_adj(self, rows, cols):
        # Generates adjacency matrix of rows*cols torus

        a1: np.ndarray = np.zeros((rows * cols, rows * cols), dtype=int)
        for i in range(rows):
            a1[i * cols, i * cols + cols - 1] = 1
            a1[i * cols + cols - 1, i * cols] = 1

        for j in range(cols):
            a1[(rows - 1) * cols + j, j] = 1
            a1[j, (rows - 1) * cols + j] = 1

        a = self.grid_adj(rows, cols) + a1
        a[a == 2] = 1
        mx = a.copy()
        return mx

    def torus_ER_adj(self, rows, cols, p):
        # Generates adjacency matrix of rows*cols + sprinkle of density p random edges
        A = self.torus_adj(rows, cols)
        U = np.random.rand(rows * cols, rows * cols)
        U = np.triu(U, k=1)
        U = U + np.transpose(U)
        U[U >= p] = 2
        U[U < p] = 1
        U[U == 2] = 0
        A = A + U
        A[A > 1] = 1
        return A

    def RW_update(self, x):
        # A = N by N matrix giving edge weights on networks
        # x = RW is currently at site x
        # stationary distribution = uniform

        A = self.A
        [N, N] = np.shape(A)
        dist_x = np.maximum(A[x, :], np.transpose(A[:, x]))
        # dist_x = A[x,:]
        #  dist_x = np.where(dist_x > 0, 1, 0)  # make 1 if positive, otherwise 0
        # (!!! The above line seem to cause disagreement b/w Pivot and Glauber chains in WAN data for
        # k1=0 and k2=1 case and other inner edge CHD cases -- 9/30/19)

        if sum(dist_x) > 0:  # this holds if the current location x of pivot is not isolated
            dist_x_new = dist_x / sum(dist_x)  # honest symmetric RW kernel
            y = np.random.choice(np.arange(0, N), p=dist_x_new)  # proposed move

            # Use MH-rule to accept or reject the move
            # Use another coin flip (not mess with the kernel) to keep the computation local and fast
            dist_y = np.maximum(A[y, :], np.transpose(A[:, y]))
            # (!!! Symmetrizing the edge weight here does not seem to affect the convergence rate for WAN data)
            # dist_y = A[y, :]
            # prop_accept = min(1, A[y, x] * sum(dist_y) / (sum(dist_x) * A[x, y]))
            prop_accept = min(1, sum(dist_x) / sum(dist_y))

            if np.random.rand() > prop_accept:
                y = x  # move to y rejected

        else:  # if the current location is isolated, uniformly choose a new location
            y = np.random.choice(np.arange(0, N))
        return y

    def Pivot_update(self, emb):
        # G = underlying simple graph
        # emb = current embedding of a path in the network
        # k1 = length of left side chain from pivot
        # updates the current embedding using pivot rule

        k1 = self.k1
        k2 = self.k2
        x0 = emb[0]  # current location of pivot
        x0 = self.RW_update(x0)  # new location of the pivot
        B = self.path_adj(k1, k2)
        #  emb_new = self.Path_sample_gen_position(x0, k1, k2)  # new path embedding
        emb_new = self.tree_sample(B, x0)  # new path embedding
        return emb_new

    def glauber_gen_update(self, B, emb):
        # A = N by N matrix giving edge weights on networks
        # emb = current embedding of the tree motif with adj mx B
        # updates the current embedding using Glauber rule

        A = self.A
        [N, N] = np.shape(A)
        [k, k] = np.shape(B)

        if k == 1:
            # emb[0] = np.random.choice(np.arange(0, N))
            # If B has no edge, conditional measure is uniform over the nodes

            '''
            For the WAN data, there is a giant connected component and the Pivot chain only explores that component. 
            In order to match the Glauber chain, we can let the single node case k1=k2=0 to behave like a RW. 
            '''
            emb[0] = self.RW_update(emb[0])
            # print('Glauber chain updated via RW')
        else:
            j = np.random.choice(np.arange(0, k))  # choose a random node to update
            nbh_in = self.indices(B[:, j], lambda x: x == 1)  # indices of nbs of j in B
            nbh_out = self.indices(B[j, :], lambda x: x == 1)  # indices of nbs of j in B

            # build distribution for resampling emb[j] and resample emb[j]
            dist = np.ones(N, dtype=int)
            for r in nbh_in:
                dist = dist * A[emb[r], :]
            for r in nbh_out:
                dist = dist * np.transpose(A[:, emb[r]])
            if sum(dist) > 0:
                dist = dist / sum(dist)
                y = np.random.choice(np.arange(0, N), p=dist)
                emb[j] = y
            else:
                emb[j] = np.random.choice(np.arange(0, N))
                print('Glauber move rejected')  # Won't happen once valid embedding is established
        return emb

    def chd_gen_mx(self, B, emb, is_Glauber=True):
        # computes B-patches of the input network G using Glauber chain to evolve embedding of B in to G
        # iterations = number of iteration
        # underlying graph = specified by A
        # B = adjacency matrix of rooted tree motif

        A = self.A
        emb2 = emb
        [N, N] = np.shape(A)
        [k,k] = np.shape(B)
        #  x0 = np.random.choice(np.arange(0, N))  # random initial location of RW
        #  emb2 = self.tree_sample(B, x0)  # initial sampling of path embedding
        hom2 = np.array([])
        hom_mx2 = np.zeros([k,k])

        for i in range(self.loc_avg_depth):
            if is_Glauber:
                emb2 = self.glauber_gen_update(B, emb2)
            else:
                emb2 = self.Pivot_update(emb2)
            # full adjacency matrix over the path motif
            a2 = np.zeros([k,k])
            for q in np.arange(k):
                for r in np.arange(k):
                    a2[q, r] = A[emb2[q], emb2[r]]

            hom_mx2 = ((hom_mx2 * i) + a2) / (i + 1)
            '''
            #  progress status
            if 100 * i / iterations % 1 == 0:
                print(i / iterations * 100)
            '''
        return hom_mx2, emb2

    def get_patches_glauber(self, B, emb, is_Glauber=True):
        # B = adjacency matrix of the motif F to be embedded into the network
        # emb = current embedding F\rightarrow G
        k = B.shape[0]
        X = np.zeros((k**2, 1))
        for i in np.arange(self.sample_size):
            # print('obtaining patch -- step %i' % i)
            Y, emb = self.chd_gen_mx(B, emb, is_Glauber)  # Y = k by k matrix, emb = one-step evolved embedding
            Y = Y.reshape(k**2, -1)
            if i == 0:
                X = Y
            else:
                X = np.append(X, Y, axis=1)  # x is class ndarray
            # print('patch_i', i)
        #  now X.shape = (k**2, sample_size)
        # print(X)
        return X, emb

    def get_single_patch_glauber(self, B, emb, is_Glauber=True):
        # B = adjacency matrix of the motif F to be embedded into the network
        # emb = current embedding F\rightarrow G
        k = B.shape[0]
        Y, emb = self.chd_gen_mx(B, emb, is_Glauber)  # k by k matrix
        X = Y.reshape(k ** 2, -1)

        #  now X.shape = (k**2, sample_size)
        # print(X)
        return X, emb

    def train_dict(self, path):
        # emb = initial embedding of the motif into the network
        print('training dictionaries from patches...')
        '''
        Trains dictionary based on patches.
        '''

        A = self.A
        [N,N] = A.shape
        B = self.path_adj(self.k1, self.k2)
        x0 = np.random.choice(np.arange(0, N))
        emb = self.tree_sample(B, x0)
        W = self.W
        At = []
        Bt = []
        Ct = []
        code = []
        errors = []
        for t in np.arange(self.MCMC_iterations):
            # print('obtaining patch in step %i started' % t)
            for i in np.arange(0):
                emb = self.glauber_gen_update(B, emb)

            X, emb = self.get_patches_glauber(B, emb)
            # print('X.size', X.shape)
            # print('obtaining patch in step %i is done' % t)

            if t == 0:
                self.nmf = Online_NMF(X,
                                      self.n_components,
                                      iterations=self.sub_iterations,
                                      batch_size=self.batch_size,
                                      alpha=self.alpha,
                                      beta=self.beta,
                                      subsample=self.ONMF_subsample)  # max number of possible patches
                self.W, self.At, self.Bt, self.Ct, self.H = self.nmf.train_dict()
                code = self.H
                error = np.trace(self.W @ self.At @ self.W.T) - 2 * np.trace(self.W @ self.Bt) + np.trace(self.Ct)
                errors.append(error)
            else:
                self.nmf = Online_NMF(X,
                                      n_components=self.n_components,
                                      iterations=self.sub_iterations,
                                      batch_size=self.batch_size,
                                      ini_dict=self.W,
                                      ini_A=self.At,
                                      ini_B=self.Bt,
                                      ini_C=self.Ct,
                                      history=self.nmf.history,
                                      alpha=None,
                                      beta=self.beta,
                                      subsample=self.ONMF_subsample)
                print('nmf.history', self.nmf.history)
                # out of "sample_size" columns in the data matrix, sample "batch_size" randomly and train the dictionary
                # for "iterations" iterations
                self.W, self.At, self.Bt, self.Ct, self.H = self.nmf.train_dict()
                code += self.H
                error = np.trace(self.W @ self.At @ self.W.T) - 2 * np.trace(self.W @ self.Bt) + np.trace(self.Ct)
                print('error', error)
                errors.append(error)
            #  progress status
            print('Current time step %i out of %i' % (t, self.MCMC_iterations))
        self.code = code
        print('code size:', code.shape)
        np.save('Network_dictionary/WAN/dict_learned' + "_" + str(self.k2) + "_" + str(self.n_components) + "_" +str(self.file_number), self.W)
        np.save('Network_dictionary/WAN/code_learned' + "_" + str(self.k2) + "_" + str(self.n_components) + "_" +str(self.file_number), self.code)
        np.save('Network_dictionary/WAN/errors_' + str(self.k2) + "_" + str(self.n_components) + "_" +str(self.file_number), errors)
        # print(self.W.shape)
        # print(self.W)

    def display_dict(self, title, save_filename):
        #  display learned dictionary
        print('W.sum', np.sum(self.W))
        W = self.W
        code = self.code  # row sum of code matrix will give importance of each dictionary patch
        n_components = W.shape[1]
        rows = np.round(np.sqrt(n_components))
        rows = rows.astype(int)
        if rows ** 2 == n_components:
            cols = rows
        else:
            cols = rows + 1

        importance = np.sum(code, axis=1)/sum(sum(code))
        idx = np.argsort(importance)
        idx = np.flip(idx)
        # print('idx', idx)
        # print('importance', importance)

        #fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(5.5, 7),
        fig, axs = plt.subplots(nrows=5, ncols=9, figsize=(7, 5),
                                    subplot_kw={'xticks': [], 'yticks': []})
        k = self.k1 + self.k2 + 1  # number of nodes in the motif F
        for ax, j in zip(axs.flat, range(n_components)):
            ax.set_xlabel('%1.2f' % importance[idx[j]], fontsize=15)  # get the largest first
            ax.xaxis.set_label_coords(0.5, -0.05)  # adjust location of importance appearing beneath patches
            ax.imshow(W.T[idx[j]].reshape(k, k), cmap="gray_r", interpolation='nearest')
            # use gray_r to make black = 1 and white = 0

        plt.suptitle(title)
        fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0)
        fig.savefig('Network_dictionary/WAN/' + save_filename + ".png")
        plt.show()

    def show_array(self, arr):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4.5),
                               subplot_kw={'xticks': [], 'yticks': []})
        ax.xaxis.set_ticks_position('bottom')
        ax.imshow(arr)
        plt.show()


    def reconstruct_network(self, recons_iter=100, alpha=0, beta=0.75, is_Glauber=True):
        print('reconstructing given network...')
        '''
        Note: For WAN data, the algorithm reconstructs the normalized WAN matrix A/np.max(A). 
        Scale the reconstructed matrix B by np.max(A) and compare with the original network. 
        '''

        A = self.A
        [N, N] = A.shape
        A_recons = np.zeros(shape=(N,N))
        A_overlap_count = np.zeros(shape=(N,N))
        B = self.path_adj(self.k1, self.k2)
        k = self.k1 + self.k2 + 1  # size of the network patch
        x0 = np.random.choice(np.arange(0, N))
        emb = self.tree_sample(B, x0)
        t0 = time()
        c = 0

        for t in np.arange(recons_iter):
            patch, emb = self.get_single_patch_glauber(B, emb, is_Glauber)
            coder = SparseCoder(dictionary=self.W.T, transform_n_nonzero_coefs=None,
                                transform_alpha=alpha, transform_algorithm='lasso_lars', positive_code=True)
            # alpha = L1 regularization parameter. alpha=2 makes all codes zero (why?)
            # This only occurs when sparse coding a single array
            code = coder.transform(patch.T)
            patch_recons = np.dot(self.W, code.T).T
            patch_recons = patch_recons.reshape(k,k)
            for x in itertools.product(np.arange(k), repeat=2):
                a = emb[x[0]]
                b = emb[x[1]]
                j = A_overlap_count[a,b].astype(float)
                if j == 0:
                    v = 0
                else:
                    v = j**(-beta)
                A_recons[a,b] = (1-v)*A_recons[a,b] + v * patch_recons[x[0], x[1]]
                # A_recons[a,b] = A_recons[a,b] + patch_recons[x[0], x[1]]
                A_overlap_count[a,b] += 1

            C = np.where(A_overlap_count>1, 1, 0)
            # progress status
            # print('Current time step %i out of %i' % (t, recons_iter))

            if t % 1000 == 0:
                print('A.norm, A_recons.norm, diff.norm', np.linalg.norm(self.A), np.linalg.norm(A_recons), np.linalg.norm(self.A - A_recons))
                print('Current time step %i out of %i' % (t, recons_iter))

        print('Reconstructed in %.2f seconds' % (time() - t0))
        np.save('Network_dictionary/WAN/twain_recons' + "_" + str(self.k2) + "_" + str(self.n_components) + str(self.file_number),
                A_recons)
        print('A.norm, A_recons.norm, diff.norm', np.linalg.norm(self.A), np.linalg.norm(A_recons),
              np.linalg.norm((self.A - A_recons)*C), np.linalg.norm(self.A - A_recons))
        print('C', np.sum(np.sum(C)))
        # only compare edges that are ever visited by the reconstruction algorithm
        return A_recons

def display_errors(k2, n_components, file_number):
    errors1 = np.load("Network_dictionary/WAN/errors_" + str(k2) + '_' + str(n_components)+ "_" +str(file_number)+'.npy')
    #errors2 = np.load("Ising/Ising_errors_" + str(T) + "_subsampling_10000.npy")
    #errors3 = np.load("Ising/Ising_errors_" + str(T) + "_subsampling_100000.npy")
    #errors4 = np.load("Ising/Ising_errors_" + str(T) + "_subsampling_500000.npy")

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
    axs.plot(500 * np.arange(len(errors1)) / len(errors1), errors1, label='')
    #axs.plot(500 * np.arange(len(errors2)) / len(errors2), errors2 / 40000, label='subsampling epoch of 10000')
    #axs.plot(500 * np.arange(len(errors3)) / len(errors3), errors3 / 40000, label='subsampling epoch of 100000')
    #axs.plot(500 * np.arange(len(errors4)) / len(errors4), errors4 / 40000, label='subsampling epoch of 500000')
    axs.legend()
    # axs.set_ylim(0,)
    # axs.set_xticks(np.arange(500))
    plt.tight_layout()
    # plt.suptitle('Dictionary learned from patches of size %d' % k, fontsize=16)
    # plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
    plt.show()

def display_recons():
    A = np.genfromtxt("Data/WAN/twain_1.txt", usecols=range(211))
    # A_recons = np.load("Network_dictionary/Wan/twain_recons_1_4512.npy")
    # A_recons = np.load("Network_dictionary/Wan/twain_recons_2_4511.npy")
    A_recons = np.load("Network_dictionary/Wan/twain_recons_3_4513.npy")
    print('A.norm, A_recons.norm, diff.norm', np.linalg.norm(A/np.max(A)), np.linalg.norm(A_recons), np.linalg.norm(A/np.max(A) - A_recons))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4.5))
    ax.xaxis.set_ticks_position('bottom')
    ax.imshow(np.log(A_recons*np.max(A) + 1))
    plt.show()

def main():
    data_rows = []
    A_recons = np.zeros(shape=(211, 211))
    A_overlap_count = np.zeros(shape=(211, 211))

    n_components = 45
    k1 = 0
    k2 = 2
    file_number = 14

    with open('Data/Wan/WAN_list.csv', 'r') as csvFile:
        # creating a csv reader object
        reader = csv.DictReader(csvFile)

        # extracting each data row one by one
        for row in reader:
            data_rows.append(row)
        # rows is a list object.

    num_files = reader.line_num - 1  # last row is fields
    sources = ["Data/WAN/" + data_rows[i]['filename'] + ".txt" for i in np.arange(num_files)]

    bar = progressbar.ProgressBar()
    for i in bar(np.arange(40,41)):
        # sources = ["Data/torus_adj.txt"]
        reconstructor = Network_Reconstructor(source=sources[i],
                                              n_components=n_components,
                                              MCMC_iterations=100,
                                              sample_size=1000,
                                              batch_size=20,
                                              ntwk_size=211,
                                              loc_avg_depth=1,
                                              sub_iterations=20,
                                              k1=k1, k2=k2,
                                              alpha=0,
                                              beta=0.75,
                                              ONMF_subsample=True,
                                              file_number = file_number)
        # For homogeneous network like the torus, setting alpha small seems to work more accurately.
        reconstructor.train_dict(sources[i])
        # reconstructor.W = np.load('Network_dictionary/WAN/dict_learned' + "_" + str(k2) + "_" + str(n_components) + "_" + str(file_number)+'.npy')
        # reconstructor.code = np.load('Network_dictionary/WAN/code_learned' + "_" + str(k2) + "_" + str(n_components) + "_" + str(file_number) + '.npy')

        ### save dictionay figures
        title = "Network dictionary patches" + "\n" + data_rows[i]['Author'] + " - " + data_rows[i]['Title']
        save_filename = data_rows[i]['filename'] + "_" + str(k2) + "_" + str(n_components) +'_'+ str(file_number)
        reconstructor.display_dict(title, save_filename)


        ### Display surrgate error for dictionary learning
        # display_errors(k2, n_components, file_number)

        '''
        reconstruct network
        '''
        A_recons = reconstructor.reconstruct_network(recons_iter=50000, alpha=0.1, beta=1)
        # reconstructor.show_array(reconstructor.A)
        # A_recons = np.load("Network_dictionary/Wan/twain_recons_2_4511.npy")
        # display_recons()


if __name__ == '__main__':
    main()

