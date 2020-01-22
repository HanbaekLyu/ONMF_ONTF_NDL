from ontf import Online_NTF
import numpy as np
from PIL import Image
from skimage.transform import downscale_local_mean
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d

from time import time

import matplotlib.pyplot as plt

DEBUG = False


class Image_Reconstructor_tensor():
    def __init__(self,
                 path,
                 n_components=100,  # number of dictionary elements -- rank
                 iterations=50,  # number of iterations for the ONTF algorithm
                 sub_iterations = 20,  # number of i.i.d. subsampling for each iteration of ONTF
                 batch_size=20,   # number of patches used in i.i.d. subsampling
                 block_iterations = 20,  # number of block optimization for factors T and H in each iteration of ONTF
                 num_patches = 1000,   # number of patches that ONTF algorithm learns from at each iteration
                 sub_num_patches = 10000,  # number of patches to optimize H after training W
                 downscale_factor=2,
                 patch_size=7,
                 patches_file='',
                 learn_joint_dict=False,
                 is_matrix=False,
                 is_stack=False,
                 is_color=True):
        '''
        batch_size = number of patches used for training dictionaries per ONTF iteration
        sources: array of filenames to make patches out of
        patches_array_filename: numpy array file which contains already read-in images
        '''
        self.path = path
        self.n_components = n_components
        self.iterations = iterations
        self.sub_iterations = sub_iterations
        self.block_iterations = block_iterations
        self.num_patches = num_patches
        self.sub_num_patches = sub_num_patches
        self.batch_size = batch_size
        self.downscale_factor = downscale_factor
        self.patch_size = patch_size
        self.patches_file = patches_file
        self.learn_joint_dict = learn_joint_dict
        self.is_matrix = is_matrix
        self.is_stack = is_stack  # if True, input data is a 3d array
        self.is_color = is_color
        self.W = np.zeros(shape=(patch_size, n_components))
        self.code = np.zeros(shape=(n_components, iterations*batch_size))

        '''
        # read in patches
        if patches_file:
            self.patches = np.load(patches_file)
        else:
            self.patches = self.read_patches(patch_size=patch_size, iterations=iterations,
                                             batch_size=batch_size, is_matrix=is_matrix)
            print(self.patches.shape)
        '''

        # read in image as array
        self.data = self.read_img_as_array()

    def read_img_as_array(self):
        '''
        Read input image as a narray
        '''

        if self.is_matrix:
            img = np.load(self.path)
            data = (img + 1) / 2  # it was +-1 matrix; now it is 0-1 matrix
        else:
            img = Image.open(self.path)
            if self.is_color:
                img = img.convert('RGB')
            else:
                img = img.convert('L')
            # normalize pixel values (range 0-1)
            data = np.asarray(img) / 255
        print('data.shape', data.shape)
        return data

    def extract_random_patches(self, second_factor=False):
        '''
        Extract 'num_patches' many random patches of given size
        color -- patch_size * patch_size * 3
        b/w -- patch_size * patch_size
        '''
        x = self.data.shape
        k = self.patch_size
        if not second_factor:
            num_patches = self.num_patches
        else:
            num_patches = self.sub_num_patches

        if len(x) == 3:  # color image
            X = np.zeros(shape=(k ** 2, 3, 1))
            for i in np.arange(num_patches):
                a = np.random.choice(x[0] - k)  # x coordinate of the top left corner of the random patch
                b = np.random.choice(x[1] - k)  # y coordinate of the top left corner of the random patch
                Y = self.data[a:a+k, b:b+k, :]
                Y = Y.reshape(k**2, 3, 1)  # size k**2 by 3
                # print('Y.shape', Y.shape)
                if i == 0:
                    X = Y
                else:
                    X = np.append(X, Y, axis=2)  # x is class ndarray
        else:  # b/w image
            X = np.zeros(shape=(k ** 2, 1, 1))
            for i in np.arange(num_patches):
                a = np.random.choice(x[0] - k)  # x coordinate of the top left corner of the random patch
                b = np.random.choice(x[1] - k)  # y coordinate of the top left corner of the random patch
                Y = self.data[a:a + k, b:b + k]
                Y = Y.reshape(k ** 2, 1, 1)  # size k**2 by 3
                # print('Y.shape', Y.shape)
                if i == 0:
                    X = Y
                else:
                    X = np.append(X, Y, axis=1)  # X is class ndarray
        return X  # X.shape = (k**2, 3, num_patches) for color, (k**2, num_patches, 1) for b/w

    def image_to_patches(self, path, patch_size=10, downscale_factor=2, is_matrix=False, is_recons=False):
        '''
        #*****

        args:
            path (string): Path and filename of input image
            patch_size (int): Pixel dimension of square patches taken of image
            color (boolean): Specifies conversion of image to RGB (True) or grayscale (False).
                Default value = false. When color = True, images in gray colorspace will still appear
                gray, but will thereafter be represented in RGB colorspace using three channels.
            downscale_factor: Specifies the extent to which the image will be downscaled. Greater values
                will result in more downscaling but faster speed. For no downscaling, use downscale_factor=1.
        returns: #***

        '''
        #open image and convert it to either RGB (three channel) or grayscale (one channel)
        if is_matrix:
            img = np.load(path)
            data = (img + 1) / 2  # it was +-1 matrix; now it is 0-1 matrix
        else:
            img = Image.open(path)
            if self.is_color:
                img = img.convert('RGB')
            else:
                img = img.convert('L')
            # normalize pixel values (range 0-1)
            data = np.asarray(img) / 255

        if DEBUG:
            print(np.asarray(img))

        patches = self.extract_random_patches()
        print('patches.shape=', patches.shape)
        return patches

    def display_dictionary(self, W, learn_joint_dict):
        k = self.patch_size
        fig, axs = plt.subplots(nrows=10, ncols=10, figsize=(6, 6),
                                subplot_kw={'xticks': [], 'yticks': []})
        for ax, i in zip(axs.flat, range(100)):
            if not learn_joint_dict:
                ax.imshow(W.T[i].reshape(k, k), cmap="gray", interpolation='nearest')
            else:
                patch = W.T[i].reshape(k, k, 3)
                ax.imshow(patch / np.max(patch))

        plt.tight_layout()
        plt.suptitle('Dictionary learned from patches of size %d' % k, fontsize=16)
        plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
        plt.show()

    def display_second_dictionary(self, H):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 2),
                                subplot_kw={'xticks': [], 'yticks': []})
        ax.imshow(H)
        plt.tight_layout()
        plt.suptitle('Dictionary learned from patches of size %d' % self.patch_size, fontsize=16)
        # plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
        plt.show()

    def display_dictionary_color_combine(self, W, H):
        k = self.patch_size
        print('W.shape', W.shape)
        print('H.shape', H.shape)
        img_dict = W[:, None, :] * H[None, :, :]
        print('img_dict.shape', img_dict.shape)
        fig, axs = plt.subplots(nrows=10, ncols=10, figsize=(6, 6),
                                subplot_kw={'xticks': [], 'yticks': []})
        for ax, i in zip(axs.flat, range(100)):
            if not self.is_color:
                ax.imshow(W.T[i].reshape(k, k), cmap="gray", interpolation='nearest')
            else:
                patch = img_dict[:,:,i].reshape(k, k, 3)
                ax.imshow(patch / np.max(patch))

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

    def train_dict(self, mode, learn_joint_dict):
        print('training dictionaries from patches along mode %i...' % mode)
        '''
        Trains dictionary based on patches from an i.i.d. sequence of batch of patches 
        mode = 0, 1, 2
        learn_joint_dict = True or False parameter
        '''
        W = self.W
        At = []
        Bt = []
        code = self.code
        for t in np.arange(self.iterations):
            X = self.extract_random_patches()
            if t == 0:
                self.ntf = Online_NTF(X, self.n_components,
                                      iterations=self.sub_iterations,
                                      sub_iterations=self.block_iterations,
                                      learn_joint_dict = learn_joint_dict,
                                      mode=mode,
                                      batch_size=self.batch_size)  # max number of possible patches
                W, At, Bt, H = self.ntf.train_dict_single()
            else:
                self.ntf = Online_NTF(X, self.n_components,
                                      iterations=self.sub_iterations,
                                      sub_iterations=self.block_iterations,
                                      batch_size=self.batch_size,
                                      ini_dict=W,
                                      ini_A=At,
                                      ini_B=Bt,
                                      learn_joint_dict=learn_joint_dict,
                                      mode=mode,
                                      history=self.ntf.history)
                # out of "sample_size" columns in the data matrix, sample "batch_size" randomly and train the dictionary
                # for "iterations" iterations
                W, At, Bt, H = self.ntf.train_dict_single()
                # code += H
            print('Current iteration %i out of %i' % (t, self.iterations))
        self.W = W
        print('dict_shape:', self.W.shape)
        print('code_shape:', self.code.shape)
        np.save('Image_dictionary/dict_learned_tensor_renoir_' + str(mode) + 'joint' + str(learn_joint_dict), self.W)
        np.save('Image_dictionary/code_learned_tensor_renoir_' + str(mode) + 'joint' + str(learn_joint_dict), self.code)
        return W

    def show_array(self, arr):
        plt.figure()
        plt.imshow(arr, cmap="gray")
        plt.show()

    def reconstruct_image(self, path, downscale_factor=None, patch_size=10, is_matrix=False):
        print('reconstructing given image...')
        if downscale_factor is None:
            downscale_factor = self.downscale_factor

        t0 = time()
        dims = self.get_downscaled_dims(path, downscale_factor, is_matrix=is_matrix)
        patches = self.image_to_patches(path, patch_size=patch_size, downscale_factor=downscale_factor,
                                        is_matrix=is_matrix, is_recons=True)
        self.ntf = Online_NTF(patches, self.n_components, self.iterations, self.batch_size)
        code = self.ntf.joint_sparse_code_tensor(patches, self.W)
        print('Reconstructed in %.2f seconds' % (time() - t0))
        patches_recons = np.dot(self.W, code).T
        patches_recons = patches_recons.reshape(patches_recons.shape[0], patch_size, patch_size)
        img_recons = reconstruct_from_patches_2d(patches_recons, (dims[0], dims[1]))
        self.show_array(img_recons)
        return code

    def reconstruct_image_color(self, path, recons_resolution=1):
        print('reconstructing given network...')
        '''
        Note: For WAN data, the algorithm reconstructs the normalized WAN matrix A/np.max(A). 
        Scale the reconstructed matrix B by np.max(A) and compare with the original network. 
        '''
        A = self.read_img_as_array(path)  # A.shape = (row, col, 3)
        A_matrix = A.reshape(-1, A.shape[1])  # (row, col, 3) --> (3row, col)
        [m, n] = A_matrix.shape
        A_recons = np.zeros(shape=A.shape)
        A_overlap_count = np.zeros(shape=(A.shape[0], A.shape[1]))
        k = self.patch_size
        t0 = time()
        c = 0
        num_rows = np.floor((A_recons.shape[0]-k)/recons_resolution).astype(int)
        num_cols = np.floor((A_recons.shape[1]-k)/recons_resolution).astype(int)

        for i in np.arange(0, A_recons.shape[0]-k, recons_resolution):
            for j in np.arange(0, A_recons.shape[1]-k, recons_resolution):
                patch = A[i:i + k, j:j + k, :]
                patch = patch.reshape((-1, 1))
                # print('patch.shape', patch.shape)
                coder = SparseCoder(dictionary=self.W.T, transform_n_nonzero_coefs=None,
                                    transform_alpha=1, transform_algorithm='lasso_lars', positive_code=True)
                # alpha = L1 regularization parameter. alpha=2 makes all codes zero (why?)
                code = coder.transform(patch.T)
                patch_recons = np.dot(self.W, code.T).T
                patch_recons = patch_recons.reshape(k, k, 3)

                # now paint the reconstruction canvas
                for x in itertools.product(np.arange(k), repeat=2):
                    c = A_overlap_count[i+x[0], j+x[1]]
                    A_recons[i+x[0], j+x[1], :] = (c*A_recons[i+x[0], j+x[1], :] + patch_recons[x[0], x[1], :])/(c+1)
                    A_overlap_count[i+x[0], j+x[1]] += 1

                # progress status
                print('reconstructing (%i, %i)th patch out of (%i, %i)' % (i/recons_resolution, j/recons_resolution, num_rows, num_cols))
        print('Reconstructed in %.2f seconds' % (time() - t0))
        print('A_recons.shape', A_recons.shape)
        np.save('Image_dictionary/img_recons_color', A_recons)
        plt.imshow(A_recons)
        return A_recons

def main():

    patch_size = 50
    sources = ["Data/renoir/" + str(n) + ".jpg" for n in np.arange(0,1)]
    for path in sources:
        reconstructor = Image_Reconstructor_tensor(path=path,
                                                   n_components=100,  # number of dictionary elements -- rank
                                                   iterations=20,  # number of iterations for the ONTF algorithm
                                                   sub_iterations=2, # number of i.i.d. subsampling for each iteration of ONTF
                                                   batch_size=100,  # number of patches used in i.i.d. subsampling
                                                   block_iterations=4, # number of block optimization for factors T and H in each iteration of ONTF
                                                   num_patches=100, # number of patches that ONTF algorithm learns from at each iteration
                                                   sub_num_patches=5000, # number of patches to optimize H after training W
                                                   downscale_factor=2,
                                                   patch_size=20,
                                                   patches_file='',
                                                   learn_joint_dict=False,
                                                   is_matrix=False,
                                                   is_stack=False,
                                                   is_color=True)
        # reconstructor.save_patches("escher_patches.npy")
        # W = reconstructor.train_dict(mode=0, learn_joint_dict=False)
        # reconstructor.display_dictionary(W)
        # H = reconstructor.train_dict(mode=1, learn_joint_dict=False)
        # reconstructor.display_second_dictionary(H)
        # W = np.load('Image_dictionary/dict_learned_tensor_renoir_combined.npy')
        # print('W.shape', W.shape)
        # reconstructor.display_dictionary_color_combine(W, H)
        # H = reconstructor.train_second_dict(dict, iter=20)
        # print('H', H)

        W = reconstructor.train_dict(mode=2, learn_joint_dict=True)
        reconstructor.display_dictionary(W, learn_joint_dict=True)

        # reconstructor.reconstruct_image("Data/escher/10.jpg", downscale_factor=1, patch_size=patch_size, is_matrix=False)
        # reconstructor.reconstruct_image("Data/renoir/0.jpg", downscale_factor=1, patch_size=patch_size, is_matrix=False)




    '''
    sources = ["Data/Ising/ising_200_trajectory_05_test" + str(n) + ".npy" for n in range(1)]
    reconstructor = Image_Reconstructor(sources=sources, patch_size=20, batch_size=20, iterations=10,
                                        downscale_factor=1, is_matrix=True, is_stack=True)
    reconstructor.train_dict(is_stack=True)
    reconstructor.reconstruct_image("Data/Ising/ising_200_5_test0.npy", downscale_factor=2, patch_size=20, is_matrix=True)
    dict = reconstructor.W  # trained dictionary
    reconstructor.display_dictionary(dict, patch_size=20)
    print(dict.shape)
    '''

if __name__ == '__main__':
    main()

