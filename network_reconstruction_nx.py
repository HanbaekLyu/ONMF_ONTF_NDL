from onmf import Online_NMF
from dyn_emb import Dyn_Emb
import numpy as np
import csv
import seaborn as sns
import progressbar
import itertools
from time import time
from numpy import linalg as LA
from sklearn.decomposition import SparseCoder
import matplotlib.pyplot as plt
import matplotlib.image
import networkx as nx
from os import listdir
from os.path import isfile, join


DEBUG = False


class Network_Reconstructor():
    def __init__(self, source, n_components=100, MCMC_iterations=500, sub_iterations=100, loc_avg_depth=1, sample_size=1000, batch_size=10, k1=1, k2=2,
                 patches_file='', is_stack=False, alpha=None, is_WAN=False, is_glauber_dict=True, is_glauber_recons=True):
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
        self.patches_file = patches_file
        self.is_stack = is_stack  # if True, input data is a 3d array
        self.W = np.zeros(shape=(k1 + k2 + 1, n_components))
        self.code = np.zeros(shape=(n_components, sample_size))
        self.alpha = alpha
        self.is_WAN = is_WAN
        self.is_glauber_dict = is_glauber_dict   ### if false, use pivot chain for dictionary learning
        self.is_glauber_recons = is_glauber_recons   ### if false, use pivot chain for reconstruction

        # read in networks
        G = self.read_networks_as_graph(source)
        self.G = G
        print('number of nodes=', len(G))

    def read_networks_as_graph(self, path):
        edgelist = np.genfromtxt(path, delimiter=',', dtype=int)
        edgelist = edgelist.tolist()
        G = nx.Graph(edgelist)
        return G

    def read_networks(self, path):
        G = nx.read_edgelist(path, delimiter=',')
        A = nx.to_numpy_matrix(G)
        A = np.squeeze(np.asarray(A))
        print(A.shape)
        # A = A / np.max(A)
        return A

    def read_WAN_networks(self, path):
        A = np.genfromtxt(path, usecols=range(211))
        A = A / np.max(A)
        return A

    def list_intersection(self, lst1, lst2):
        temp = set(lst2)
        lst3 = [value for value in lst1 if value in temp]
        return lst3

    def np2nx(self, x):
        ### Gives bijection from np array node ordering to G.node()
        G = self.G
        a = np.asarray([v for v in G])
        return a[x]

    def nx2np(self, y):
    ### Gives bijection from G.node() to nx array node ordering
        G = self.G
        a = np.asarray([v for v in G])
        return np.min(np.where(a == y))

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

        G = self.G
        N = len(G)
        k = np.shape(B)[0]
        emb = np.array([x])  # initialize path embedding

        if sum(sum(B)) == 0:  # B is a set of isolated nodes
            y = np.random.randint(N, size=(1, k-1))
            y = y[0]  # juts to make it an array
            emb = np.hstack((emb, y))
        else:
            for i in np.arange(1, k):
                j = self.find_parent(B, i)
                nbs_j = np.asarray([i for i in G[emb[j]]])
                if len(nbs_j) > 0:
                    y = np.random.choice(nbs_j)
                else:
                    y = emb[j]
                    print('tree_sample_failed:isolated')
                emb = np.hstack((emb, y))
        # print('emb', emb)
        return emb

    def glauber_gen_update(self, B, emb):
        # A = N by N matrix giving edge weights on networks
        # emb = current embedding of the tree motif with adj mx B
        # updates the current embedding using Glauber rule

        G = self.G
        k = np.shape(B)[0]

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
            cmn_nbs = [i for i in G]
            for r in nbh_in:
                nbs_r = [i for i in G[emb[r]]]
                cmn_nbs = list(set(cmn_nbs) & set(nbs_r))
            for r in nbh_out:
                nbs_r = [i for i in G[emb[r]]]
                cmn_nbs = list(set(cmn_nbs) & set(nbs_r))
            if len(cmn_nbs) > 0:
                y = np.random.choice(np.asarray(cmn_nbs))
                emb[j] = y
            else:
                emb[j] = np.random.choice(np.asarray([i for i in G]))
                print('Glauber move rejected')  # Won't happen once valid embedding is established
        return emb

    def RW_update(self, x):
        # G = simple graph
        # x = RW is currently at site x
        # stationary distribution = uniform

        G = self.G
        nbs_x = np.asarray([i for i in G[x]])  # array of neighbors of x in G
        #  dist_x = np.where(dist_x > 0, 1, 0)  # make 1 if positive, otherwise 0
        # (!!! The above line seem to cause disagreement b/w Pivot and Glauber chains in WAN data for
        # k1=0 and k2=1 case and other inner edge CHD cases -- 9/30/19)

        if len(nbs_x) > 0:  # this holds if the current location x of pivot is not isolated
            y = np.random.choice(nbs_x)  # choose a uniform element in nbs_x

            # Use MH-rule to accept or reject the move
            # Use another coin flip (not mess with the kernel) to keep the computation local and fast
            nbs_y = np.asarray([i for i in G[y]])
            prop_accept = min(1, len(nbs_x)/len(nbs_y))

            if np.random.rand() > prop_accept:
                y = x  # move to y rejected

        else:  # if the current location is isolated, uniformly choose a new location
            y = np.random.choice(np.asarray([i for i in G]))
        return y

    def pivot_acceptance_prob(self, x, y):
        # approximately compute acceptance probability for moving the pivot of B from x to y
        G = self.A
        k = self.k1 + self.k2 + 1
        nbs_x = np.asarray([i for i in G[x]])
        nbs_y = np.asarray([i for i in G[y]])
        accept_prob = len(nbs_x) ** (k - 2) / len(nbs_y) ** (k - 2)  # to be modified

        return accept_prob

    def RW_update_gen(self, x):
        # A = N by N matrix giving edge weights on networks
        # x = RW is currently at site x
        # Acceptance prob will be computed by conditionally embedding the rest of B pivoted at x and y

        G = self.G
        nbs_x = np.asarray([i for i in G[x]])

        if len(nbs_x) > 0:  # this holds if the current location x of pivot is not isolated
            y = np.random.choice(nbs_x)  # proposed move

            accept_prob = self.pivot_acceptance_prob(x, y)
            if np.random.rand() > accept_prob:
                y = x  # move to y rejected

        else:  # if the current location is isolated, uniformly choose a new location
            y = np.random.choice(np.asarray([i for i in G]))
        return y

    def Path_sample_gen_position(self, x):
        # A = N by N matrix giving edge weights on networks
        # number of nodes in path
        # samples k1 nodes to the left and k2 nodes to the right of pivot x

        G = self.G
        k1 = self.k1
        k2 = self.k2
        emb = np.array([x]) # initialize path embedding

        for i in np.arange(0, k2):
            nbs_emb_i = np.asarray([j for j in G[emb[i]]])
            if len(nbs_emb_i) > 0:
                y1 = np.random.choice(nbs_emb_i)
            else:
                y1 = emb[i]
                # if the new location of pivot makes embedding the path impossible,
                # just contract the path onto the pivot
            emb = np.hstack((emb, [y1]))

        a = np.array([x])
        b = np.matlib.repmat(a, 1, k1+1)
        b = b[0, :]
        emb = np.hstack((b, emb[1:k2+1]))

        for i in np.arange(0, k1):
            nbs_emb_i = np.asarray([j for j in G[emb[i]]])
            if len(nbs_emb_i) > 0:
                y2 = np.random.choice(nbs_emb_i)
                emb[i+1] = y2
            else:
                emb[i + 1] = emb[i]

        return emb

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

    def chd_gen_mx(self, B, emb, iterations=1000, is_glauber=True):
        # computes B-patches of the input network G using Glauber chain to evolve embedding of B in to G
        # iterations = number of iteration
        # underlying graph = specified by A
        # B = adjacency matrix of rooted tree motif

        G = self.G
        emb2 = emb
        N = len(G)
        k = B.shape[0]
        #  x0 = np.random.choice(np.arange(0, N))  # random initial location of RW
        #  emb2 = self.tree_sample(B, x0)  # initial sampling of path embedding
        hom2 = np.array([])
        hom_mx2 = np.zeros([k, k])

        for i in range(iterations):
            if is_glauber:
                emb2 = self.glauber_gen_update(B, emb2)
            else:
                emb2 = self.Pivot_update(emb2)

            # full adjacency matrix over the path motif
            a2 = np.zeros([k, k])
            for q in np.arange(k):
                for r in np.arange(k):
                    a2[q, r] = int(G.has_edge(emb2[q], emb2[r]) == True)

            hom_mx2 = ((hom_mx2 * i) + a2) / (i + 1)
            '''
            #  progress status
            if 100 * i / iterations % 1 == 0:
                print(i / iterations * 100)
            '''
        return hom_mx2, emb2

    def get_patches_glauber(self, B, emb):
        # B = adjacency matrix of the motif F to be embedded into the network
        # emb = current embedding F\rightarrow G
        k = B.shape[0]
        X = np.zeros((k ** 2, 1))
        for i in np.arange(self.sample_size):
            Y, emb = self.chd_gen_mx(B, emb, iterations=1, is_glauber=self.is_glauber_dict)  # k by k matrix
            Y = Y.reshape(k ** 2, -1)
            if i == 0:
                X = Y
            else:
                X = np.append(X, Y, axis=1)  # x is class ndarray
        #  now X.shape = (k**2, sample_size)
        # print(X)
        return X, emb

    def get_single_patch_glauber(self, B, emb):
        # B = adjacency matrix of the motif F to be embedded into the network
        # emb = current embedding F\rightarrow G
        k = B.shape[0]
        Y, emb = self.chd_gen_mx(B, emb, iterations=1, is_glauber=self.is_glauber_recons)  # k by k matrix
        X = Y.reshape(k ** 2, -1)

        #  now X.shape = (k**2, sample_size)
        # print(X)
        return X, emb

    def train_dict(self, filename):
        # emb = initial embedding of the motif into the network
        print('training dictionaries from patches...')
        '''
        Trains dictionary based on patches.
        '''

        G = self.G
        B = self.path_adj(self.k1, self.k2)
        x0 = np.random.choice(np.asarray([i for i in G]))
        emb = self.tree_sample(B, x0)
        W = self.W
        At = []
        Bt = []
        code = self.code
        for t in np.arange(self.MCMC_iterations):
            X, emb = self.get_patches_glauber(B, emb)
            if t == 0:
                self.nmf = Online_NMF(X, self.n_components,
                                      iterations=self.sub_iterations,
                                      batch_size=self.batch_size,
                                      alpha=self.alpha)  # max number of possible patches
                W, At, Bt, H = self.nmf.train_dict()
            else:
                self.nmf = Online_NMF(X, self.n_components,
                                      iterations=self.sub_iterations,
                                      batch_size=self.batch_size,
                                      ini_dict=W,
                                      ini_A=At,
                                      ini_B=Bt,
                                      alpha=self.alpha,
                                      history=self.nmf.history)
                # out of "sample_size" columns in the data matrix, sample "batch_size" randomly and train the dictionary
                # for "iterations" iterations
                W, At, Bt, H = self.nmf.train_dict()
                code += H
            #  progress status
            # if 100 * t / self.MCMC_iterations % 1 == 0:
            #    print(t / self.MCMC_iterations * 100)
            print('Current iteration %i out of %i' % (t, self.MCMC_iterations))
        self.W = W
        self.code = code
        print('code size:', code.shape)
        np.save(
            'Network_dictionary/Facebook/dict_learned' + "_" + str(self.k1) + str(self.k2) + "_" + filename, W)
        np.save(
            'Network_dictionary/Facebook/code_learned' + "_" + str(self.k1) + str(self.k2) + "_" + filename, code)
        # print(self.W.shape)
        # print(self.W)

    def display_dict(self, title, save_filename):
        #  display learned dictionary
        W = self.W
        code = self.code  # row sum of code matrix will give importance of each dictionary patch
        n_components = W.shape[1]
        rows = np.round(np.sqrt(n_components))
        rows = rows.astype(int)
        if rows ** 2 == n_components:
            cols = rows
        else:
            cols = rows + 1


        # fig, axs = plt.subplots(nrows=5, ncols=9, figsize=(7, 5),
        fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(5, 5),
                                    subplot_kw={'xticks': [], 'yticks': []})
        k = self.k1 + self.k2 + 1  # number of nodes in the motif F
        for ax, j in zip(axs.flat, range(n_components)):
            # importance = sum(code[j, :])/sum(sum(code))
            # ax.set_xlabel('%1.2f' % importance, fontsize=15)
            # ax.xaxis.set_label_coords(0.5, -0.05)  # adjust location of importance appearing beneath patches
            ax.imshow(W.T[j].reshape(k, k), cmap="gray_r", interpolation='nearest')
            # use gray_r to make black = 1 and white = 0

        plt.suptitle(title)
        fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0)
        fig.savefig('Network_dictionary/Facebook/' + save_filename + ".png")
        # plt.show()

    def show_array(self, arr):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4.5),
                               subplot_kw={'xticks': [], 'yticks': []})
        ax.xaxis.set_ticks_position('bottom')
        ax.imshow(arr)
        plt.show()

    def show_cov(self):
        ### Computes and displays covariance matrix of the learned dictionary

        code = self.code
        cov = np.cov(code)
        cov_normalized = cov/np.trace(cov)

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4.5),
                               subplot_kw={'xticks': [], 'yticks': []})
        ax.xaxis.set_ticks_position('bottom')
        ax.imshow(cov_normalized)
        plt.colorbar()
        plt.show()
        return cov_normalized

    def reconstruct_network(self, recons_iter=100):
        print('reconstructing given network...')
        '''
        Networkx version of the reconstruction algorithm
        Note: For WAN data, the algorithm reconstructs the normalized WAN matrix A/np.max(A). 
        Scale the reconstructed matrix B by np.max(A) and compare with the original network. 
        '''

        G = self.G
        G_recons = nx.DiGraph()
        G_overlap_count = nx.DiGraph()
        G_recons.add_nodes_from([v for v in G])
        G_overlap_count.add_nodes_from([v for v in G])
        B = self.path_adj(self.k1, self.k2)
        k = self.k1 + self.k2 + 1  # size of the network patch
        x0 = np.random.choice(np.asarray([i for i in G]))
        emb = self.tree_sample(B, x0)
        t0 = time()
        c = 0

        for t in np.arange(recons_iter):
            patch, emb = self.get_single_patch_glauber(B, emb)
            coder = SparseCoder(dictionary=self.W.T,
                                transform_n_nonzero_coefs=None,
                                transform_alpha=0,
                                transform_algorithm='lasso_lars',
                                positive_code=True)
            # alpha = L1 regularization parameter. alpha=2 makes all codes zero (why?)
            # This only occurs when sparse coding a single array
            code = coder.transform(patch.T)
            patch_recons = np.dot(self.W, code.T).T
            patch_recons = patch_recons.reshape(k,k)

            for x in itertools.product(np.arange(k), repeat=2):
                a = emb[x[0]]
                b = emb[x[1]]
                ind = int(G_overlap_count.has_edge(a,b) == True)
                if ind == 1:
                    # print(G_recons.edges)
                    # print('ind', ind)
                    j = G_overlap_count[a][b]['weight']
                    new_edge_weight = (j * G_recons[a][b]['weight'] + patch_recons[x[0], x[1]]) / (j + 1)
                else:
                    j = 0
                    new_edge_weight = patch_recons[x[0], x[1]]

                G_recons.add_edge(a, b, weight=new_edge_weight)
                G_overlap_count.add_edge(a, b, weight=j + 1)
            ### Only repaint upper-triangular


            # progress status
            # print('iteration %i out of %i' % (t, recons_iter))
            if 1000 * t / recons_iter % 1 == 0:
                print(t / recons_iter * 100)

        ### Round the continuum-valued Recons matrix into 0-1 matrix.
        G_recons_simple = nx.Graph()
        # edge_list = [edge for edge in G_recons.edges]
        for edge in G_recons.edges:
            [a, b] = edge
            conti_edge_weight = G_recons[a][b]['weight']
            binary_edge_weight = np.round(conti_edge_weight)
            if binary_edge_weight > 0:
                G_recons_simple.add_edge(a, b)

        print('Reconstructed in %.2f seconds' % (time() - t0))
        return G_recons_simple

    def compute_recons_accuracy(self, path):
        ### Compute reconstruction error
        G = self.G
        G_recons = self.read_networks_as_graph(path)
        G_recons.add_nodes_from(G.nodes)
        H = nx.intersection(G, G_recons)
        recons_accuracy = len(H.edges) / len(G.edges)

        print('# edges of original ntwk=', len(G.edges))
        print('# edges of reconstructed ntwk=', len(G_recons.edges))
        print('reconstruction accuracy=', recons_accuracy)
        return H, recons_accuracy

    def compute_A_recons(self, path):
        ### Compute reconstruction error
        G_recons = self.read_networks_as_graph(path)
        G_recons.add_nodes_from(self.G.nodes)
        A_recons = nx.to_numpy_matrix(G_recons, nodelist=self.G.nodes)
        ### Having "nodelist=G.nodes" is CRUCIAL!!!
        ### Need to use the same node ordering between A and G for A_recons and G_recons.
        return A_recons

def main():
    ### set motif arm lengths
    k1 = 0
    k2 = 20
    n_components = 25

    ### read/save individual facebook data
    # school = "UCLA26.txt"
    # directory = "Data/Facebook/SchoolDataPythonFormat/"
    # path = directory + school
    # path = "Data/WAN/abbott_1.txt"
    # path = "Data/torus_adj.txt"

    ### Create list of file names
    # myfolder = "Data/WAN/sub_WAN"
    # myfolder = "Data/Facebook/SchoolDataPythonFormat/sub_fb_networks"
    # onlyfiles = [f for f in listdir(myfolder) if isfile(join(myfolder, f))]
    onlyfiles = ['MIT8.txt']
    # onlyfiles.remove('desktop.ini')

    for school in onlyfiles:
        directory = "Data/Facebook/SchoolDataPythonFormat/"
        # directory = "Data/WAN/sub_WAN/"
        path = directory + school
        print('Currently learning dictionary patches from ' + school)

        reconstructor = Network_Reconstructor(source=path,
                                              n_components=n_components,  # num of dictionaries
                                              MCMC_iterations=200,   # MCMC steps (macro, grow with size of ntwk)
                                              loc_avg_depth=1,    # keep it at 1
                                              sample_size=1000,  # number of patches in a single batch
                                              batch_size=20,  # number of columns used to train dictionary
                                                              # within a single batch step (keep it)
                                              sub_iterations=100,  # number of iterations of the
                                                                    # sub-batch learning (keep it)
                                              k1=k1, k2=k2,  # left and right arm lengths
                                              alpha=1,  # parameter for sparse coding, higher for stronger smoothing
                                              is_WAN=False,   # keep it false for FB networks
                                              is_glauber_dict=True,  # keep true to use Glauber chain for dict. learning
                                              is_glauber_recons=False)  # keep false to use Pivot chain for recons.

        # For homogeneous network like the torus, setting alpha small seems to work more accurately.
        reconstructor.train_dict(filename=school + '_' + str(n_components))
        # W = reconstructor.W  # trained dictionary

        school = 'MIT8.txt'   ### Origination of dictionary
        #reconstructor.W = np.load('Network_dictionary/Facebook/dicts_learned/dict_learned' + str(school) + '_k2_10_MCMCiter_250_alpha_1_samSz_1000.npy')
        # reconstructor.W = np.load('Network_dictionary/Facebook/dict_learned' + "_" + str(k1) + str(k2) + "_" + school + '_' + str(n_components) + '.npy')
        # reconstructor.W = np.load('Image_dictionary/dict_learned_21.npy')
        # reconstructor.code = np.load('Network_dictionary/Facebook/code_learned' + "_" + str(k1) + str(k2) + "_" + school + '_' + str(n_components) + '.npy')

        ### save dictionaytrain_dict figures
        title = str(k1+k2+1) + "by" + str(k1+k2+1) + "Network dictionary patches" + "\n" + school + " facebook network"
        save_filename = "dict_plot_" + school + "_" + str(k1) + str(k2) + '_' + str(n_components)
        reconstructor.display_dict(title, save_filename)

        '''
        # progress status
        if 100 * i / num_files % 1 == 0:
            print(i / num_files * 100)
        '''


        ### reconstruct network

        # school = 'Brown11.txt'
        # G_recons = reconstructor.reconstruct_network(recons_iter=100000)
        # np.save('Network_dictionary/Facebook/' + school + '_recons_100_self_pivot_mx', G_recons)

        '''            
        path_recons = 'Network_dictionary/Facebook/' + school + '_recons_UCLA_pivot' + '_' + str(n_components) + '.txt'
        nx.write_edgelist(G_recons,
                          path=path_recons,
                          data=False,
                          delimiter=",")
        print('Reconstruction Saved')
        # G_common, recons_accuracy = reconstructor.compute_recons_accuracy(path_recons)
        '''

if __name__ == '__main__':
    main()

