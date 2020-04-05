from onmf import Online_NMF
import itertools
import numpy as np
from PIL import Image
from skimage.transform import downscale_local_mean
from ising_simulator import ising_update
import matplotlib.pyplot as plt
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d

DEBUG = False


class Ising_Reconstructor():
    def __init__(self,
                 n_components=100,
                 lattice_size=200,
                 ising_iterations=500,
                 temperature=0.5,
                 ising_subsampling_steps=100,
                 sub_iterations=20,
                 num_patches=1000,
                 batch_size=20,
                 downscale_factor=2,
                 patch_size=20,
                 beta=0.5):
        '''
        batch_size = number of patches used for training dictionaries per ONMF iteration
        sources: array of filenames to make patches out of
        patches_array_filename: numpy array file which contains already read-in images
        '''
        self.n_components = n_components
        self.lattice_size = lattice_size
        self.temperature = temperature
        self.ising_iterations = ising_iterations
        self.ising_subsampling_steps = ising_subsampling_steps
        self.sub_iterations = sub_iterations
        self.num_patches = num_patches
        self.batch_size = batch_size
        self.downscale_factor = downscale_factor
        self.beta=beta
        self.patch_size = patch_size
        self.W = np.zeros(shape=(patch_size, n_components))
        self.data = np.zeros((3,3))

    def extract_random_patches(self):
        '''
        Extract 'num_patches' many random patches of given size
        color -- 3 patch_size * patch_size (color pixels are flattened)
        b/w -- patch_size * patch_size
        '''
        x = self.data.shape
        k = self.patch_size

        X = np.zeros(shape=(k ** 2, 1, 1))
        for i in np.arange(self.num_patches):
            a = np.random.choice(x[0] - k)  # x coordinate of the top left corner of the random patch
            b = np.random.choice(x[1] - k)  # y coordinate of the top left corner of the random patch
            Y = self.data[a:a + k, b:b + k]
            Y = Y.reshape(k ** 2, 1)  # size k**2 by 1
            # print('Y.shape', Y.shape)
            if i == 0:
                X = Y
            else:
                X = np.append(X, Y, axis=1)  # X is class ndarray
        return X  # X.shape = (k**2, 1)

    def display_dictionary(self, W):
        k = self.patch_size
        fig, axs = plt.subplots(nrows=10, ncols=10, figsize=(6,6),
                                subplot_kw={'xticks':[], 'yticks': []})
        for ax, i in zip(axs.flat, range(100)):
            ax.imshow(W.T[i].reshape(k, k), cmap="gray", interpolation='nearest')

        plt.tight_layout()
        plt.suptitle('Dictionary learned from patches of size %d' % k, fontsize=16)
        plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
        plt.show()

    def get_downscaled_dims(self, path, downscale_factor=None, is_matrix=False):
        # need to put is_matrix arg at the end to avoid error (don't know why)
        if downscale_factor is None:
            downscale_factor = self.downscale_factor

        if not is_matrix:
            img = Image.open(path)
            img = np.asarray(img.convert('L'))
        else:
            img = np.load(path)

        data = downscale_local_mean(img, (downscale_factor, downscale_factor))
        return data.shape[0], data.shape[1]

    def show_array(self, arr):
        plt.figure()
        plt.imshow(arr, cmap="gray")
        plt.show()

    def ising_mcmc_learning(self):
        ### Ising model simulator and dictionary learning along mcmc trajectory
        # lattice = np.random.choice([1, -1], size=(self.lattice_size, self.lattice_size))  ### initial spin configuration
        lattice = np.load("Ising/Ising_trajectory_0.520.npy")[-1]
        burn_in_period = 0
        lattice, energies, spins = ising_update(lattice,
                                                nsteps=burn_in_period,
                                                J=1, H=0,
                                                T=self.temperature)

        # self.show_array(lattice)
        trajectory = []
        errors = []

        ### initialize training
        self.data = lattice
        X = self.extract_random_patches()
        self.nmf = Online_NMF(X,
                              n_components=self.n_components,
                              iterations=self.sub_iterations,
                              batch_size=self.batch_size,
                              ini_dict=None,
                              ini_A=None,
                              ini_B=None,
                              ini_C=None,
                              history=0,
                              alpha=None,
                              beta=self.beta)
        W, At, Bt, Ct, H = self.nmf.train_dict()
        self.W = W
        self.At = At
        self.Bt = Bt
        self.Ct = Ct
        self.H = H
        error = np.trace(W @ At @ W.T) - 2*np.trace(W @ Bt) + np.trace(Ct)
        errors.append(error)

        dict = np.expand_dims(self.W, axis=2)
        # dict = np.zeros(shape=(self.patch_size**2, self.n_components, self.ising_iterations))  # preallocate
        print('dict_shape', dict.shape)
        print('ising_iterations', self.ising_iterations)

        # trajectory = np.zeros(shape=(self.lattice_size, self.lattice_size, self.ising_iterations))  # preallocate
        for time in np.arange(self.ising_iterations):  ### along the time dimension
            ### update Ising spin configuation
            # lattice, energies, spins = ising_update(lattice, nsteps=self.ising_subsampling_steps, J=1, H=0, T=self.temperature)
            # trajectory.append(lattice)  ### update MCMC trajectory

            self.data = lattice
            X = self.extract_random_patches()
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
                                  beta=self.beta)
            print('nmf.history', self.nmf.history)
            # out of "sample_size" columns in the data matrix, sample "batch_size" randomly and train the dictionary
            # for "iterations" iterations
            self.W, self.At, self.Bt, self.Ct, self.H = self.nmf.train_dict()
            error = np.trace(self.W @ self.At @ self.W.T) - 2 * np.trace(self.W @ self.Bt) + np.trace(self.Ct)
            print('error %1.2f at iteration %i' % (error, time))
            errors.append(error)

            dict = np.concatenate((dict, np.expand_dims(self.W, axis=2)), axis=2)
            print('dict_shape', dict.shape)
            print('Current time step %i out of %i' % (time, self.ising_iterations))

        self.W = dict[:,:,-1]
        print('dict_shape:', self.W.shape)
        # np.save('Ising/dict_learned_ising_' + str(self.temperature)+'_subsampling_'+str(self.ising_subsampling_steps), self.W)
        # np.save('Ising/dict_learned_ising_' + str(self.temperature)+'_subsampling_'+str(self.ising_subsampling_steps), dict)
        # np.save('Ising/Ising_trajectory_' + str(self.temperature)+'_subsampling_'+str(self.ising_subsampling_steps), trajectory)
        # np.save('Ising/Ising_errors_' + str(self.temperature) + '_subsampling_' + str(self.ising_subsampling_steps), errors)

        return trajectory, dict, errors

    def array_to_patches(self, config):
        # img = np.load(path)
        img = config
        data = (img + 1) / 2  # it was +-1 matrix; now it is 0-1 matrix
        patches = extract_patches_2d(data, (self.patch_size, self.patch_size))
        patches_flat = patches.reshape(len(patches), -1).T
        print(patches_flat.shape)
        return patches_flat

    def reconstruct_config(self, config, patch_size=20):
        print('reconstructing given configuration...')

        patches = self.array_to_patches(config)
        nmf = Online_NMF(patches)
        code = nmf.sparse_code(patches, self.W)
        patches_recons = np.dot(self.W, code).T
        patches_recons = patches_recons.reshape(patches_recons.shape[0], patch_size, patch_size)
        img = config
        img_recons = reconstruct_from_patches_2d(patches_recons, (img.shape[0], img.shape[1]))
        self.show_array(img_recons)
        return code

def display_errors(T):
        errors1 = np.load("Ising/Ising_errors_" + str(T) + "_subsampling_1000.npy")
        errors2 = np.load("Ising/Ising_errors_" + str(T) + "_subsampling_10000.npy")
        errors3 = np.load("Ising/Ising_errors_" + str(T) + "_subsampling_100000.npy")
        errors4 = np.load("Ising/Ising_errors_" + str(T) + "_subsampling_500000.npy")

        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(4,4))
        axs.plot(500*np.arange(len(errors1)) / len(errors1), errors1/40000, label='subsampling epoch of 1000')
        axs.plot(500*np.arange(len(errors2)) / len(errors2), errors2/40000, label='subsampling epoch of 10000')
        axs.plot(500*np.arange(len(errors3)) / len(errors3), errors3/40000, label='subsampling epoch of 100000')
        axs.plot(500 * np.arange(len(errors4)) / len(errors4), errors4/40000, label='subsampling epoch of 500000')
        axs.legend()
        # axs.set_ylim(0,)
        # axs.set_xticks(np.arange(500))
        plt.tight_layout()
        # plt.suptitle('Dictionary learned from patches of size %d' % k, fontsize=16)
        # plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
        plt.show()

def ising_sim():
    reconstructor = Ising_Reconstructor(lattice_size=200,
                                        temperature=5,
                                        ising_iterations=1,
                                        ising_subsampling_steps=500000,
                                        sub_iterations=20,
                                        batch_size=50,
                                        num_patches=1000,
                                        patch_size=20,
                                        downscale_factor=1,
                                        beta=1)
    trajectory, dict, errors = reconstructor.ising_mcmc_learning()
    # dict = np.load("Ising/dict_learned_ising_5_subsampling_500000.npy")
    # trajectory = np.load("Ising/Ising_trajectory_5_subsampling_500000.npy")
    reconstructor.display_dictionary(dict[:, :, -1])
    # reconstructor.W = dict[:,:,-1]
    # reconstructor.show_array(trajectory[-1,:,:])
    # display_errors(T=5)
    # reconstructor.reconstruct_config(trajectory[-1,:,:], patch_size=20)

def main():

    ising_sim()


if __name__ == '__main__':
    main()

