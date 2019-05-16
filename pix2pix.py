from __future__ import print_function, division
import scipy
import tensorflow as tf
import keras.backend as K
from keras.datasets import mnist
# from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, Maximum, Add
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD, Adadelta
import datetime
import matplotlib.pyplot as plt
import sys
# from data_loader import DataLoader
import numpy as np
import os
from scipy import misc
import cv2

def structural_loss(y_true,y_pred):
    # img1 = tf.image.rgb_to_grayscale(y_true)
    # img2 = tf.image.rgb_to_grayscale(y_pred)
    # print(img1)
    # st_loss = zero
    zero = tf.constant(-1, dtype=tf.float32)
    # st_loss = zero
    b = K.not_equal(y_true, zero)
    fb = K.cast(b, dtype='float32')
    nb = tf.reduce_sum(fb)
    c  = K.not_equal(y_pred, zero)
    # cn = K.equal(y_pred, zero)
    fc = K.cast(c, dtype='float32')
    nc = tf.reduce_sum(fc)

    d = tf.logical_or(b,c)
    fd = K.cast(d, dtype='float32')
    nd = tf.reduce_sum(fd)

    e = tf.logical_and(b, c)
    fe = K.cast(e, dtype='float32')
    ne = tf.reduce_sum(fe)

    term1 = tf.divide(nd,(nb+1e-08))
    term2 = tf.divide(nb,(ne+1e-08))


    st_loss = tf.reduce_mean(tf.add(term1,term2))
    # fake_where = tf.not_equal(y_pred, zero)
    # true_where = tf.zeros(fake_where.shape)
    # # K.greater(y_pred,zero)
    # true_where = tf.not_equal(y_true, zero)
    #
    # true_where = tf.cast(true_where, tf.int32)
    # fake_where = tf.cast(fake_where, tf.int32)
    # print(true_where)
    # print(fake_where)
    # print(true_where.shape)
    # print(y_pred.shape)
    # ones_array = tf.ones(y_pred.shape)
    # bc_loss = K.binary_crossentropy(y_true,y_pred)
    # lossM = tf.bitwise.bitwise_xor(true_where,fake_where)
    # print(lossM.type())
    # mse_loss = K.mean(K.square(y_true - y_pred))
    # st_loss = K.abs((ssim_loss + mse_loss))
    # st_loss = tf.reduce_sum(true_where,fake_where)

    return st_loss


def imageLoader(files1, batch_start=0, batch_size=1):
    L = len(files1)

    # this line is just to make the generator infinite, keras needs that
    while True:

        # batch_start = 0
        batch_end = batch_start + batch_size

        while batch_start < L:
            limit = min(batch_end, L)
            # print type(files1[batch_start:limit])
            X = readImages(files1[batch_start:limit], '405_ann')
            Y = readImagesEdge(files1[batch_start:limit], '405_imgs')
            Z = readImages(files1[batch_start:limit], '405_simp_v2')
            # batch_start += batch_size
            # batch_end += batch_size
            yield (X, Y, Z)  # a tuple with two numpy arrays with batch_size samples

def imageLoaderReturn(files1, batch_start=0, batch_size=1):
    L = len(files1)

    # this line is just to make the generator infinite, keras needs that
    while True:

        # batch_start = 0
        batch_end = batch_start + batch_size

        while batch_start < L:
            limit = min(batch_end, L)
            # print type(files1[batch_start:limit])
            # print(files1[batch_start:limit])
            X = readImages(files1[batch_start:limit], '405_ann')
            Y = readImagesEdge(files1[batch_start:limit], '405_imgs')
            Z = readImages(files1[batch_start:limit], '405_simp_v2')
            # batch_start += batch_size
            # batch_end += batch_size

            return X, Y, Z


def readImagesEdge(file1, name):
    L = len(file1)
    X = []
    filePath = '/home/samik/Documents/P/'
    # filePath = '/home/samik/mnt/gpu3/mnt/disk128/main/training_data/training_data/STP/'
    for f in file1:
        img = misc.imread(filePath + name + "/" + f,0)
        # print(np.unique(img))
        # plt.imshow(img)
        # plt.show()
        # img = cv2.Canny(img,1,10)
        # plt.imshow(img)
        # plt.show()
        img = (img.astype(np.float32) - 127.5) / 127.5
        # img = (img - np.min(img))/[(np.max(img) - np.min(img))+1e-08]
        # print np.max(img)
        # print np.min(img)
        # print "+++++++++"
        # np.expand_dims(img,axis=3)
        X.append(img)

    X_arr = np.asarray(X)
    # print X_arr.shape
    # np.expand_dims(X_arr, axis=3)
    X_arr = X_arr[..., np.newaxis]
    # X_arr.expand_dims(1)
    # print X_arr.shape

    return X_arr


def readImages(file1, name):
    L = len(file1)
    X = []
    filePath = '/home/samik/Documents/P/'
    # filePath = '/home/samik/mnt/gpu3/mnt/disk128/main/training_data/training_data/STP/'
    for f in file1:
        img = misc.imread(filePath + name + "/" + f)
        # plt.imshow(img)
        # plt.show()
        img = (img.astype(np.float32) - 127.5) / 127.5
        # img = (img - np.min(img))/[(np.max(img) - np.min(img))+1e-08]
        # print np.max(img)
        # print np.min(img)
        # print "+++++++++"
        # np.expand_dims(img,axis=3)
        X.append(img)

    X_arr = np.asarray(X)
    # print X_arr.shape
    # np.expand_dims(X_arr, axis=3)
    X_arr = X_arr[..., np.newaxis]
    # X_arr.expand_dims(1)
    # print X_arr.shape

    return X_arr

class Pix2Pix():
    def __init__(self):
        # Input shape
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Configure data loader
        # self.dataset_name = 'facades'
        # self.data_loader = DataLoader(dataset_name=self.dataset_name,
        #                               img_res=(self.img_rows, self.img_cols))


        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        optimizer = SGD(0.00002,momentum = 0.9)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer)

        #-------------------------
        # Construct Computational
        #   Graph of Generator
        #-------------------------

        # Build the generator
        self.generator = self.build_generator()

        # Input images and their conditioning images
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)
        img_C = Input(shape=self.img_shape)

        # By conditioning on B generate a fake version of A
        fake_A = self.generator([img_B, img_C])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_A, img_B, img_C])

        self.combined = Model(inputs=[img_A, img_B, img_C], outputs=[valid, fake_A])
        self.combined.compile(loss_weights=[1, 100],
                              loss=['binary_crossentropy', structural_loss],
                              optimizer=optimizer)

    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d01 = Input(shape=self.img_shape)
        d02 = Input(shape=self.img_shape)
        d0 = Maximum()([d01, d02])

        # Downsampling
        d1 = conv2d(d0, self.gf, bn=False)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)
        d5 = conv2d(d4, self.gf*8)
        d6 = conv2d(d5, self.gf*8)
        d7 = conv2d(d6, self.gf*8)

        # Upsampling
        u1 = deconv2d(d7, d6, self.gf*8)
        u2 = deconv2d(u1, d5, self.gf*8)
        u3 = deconv2d(u2, d4, self.gf*8)
        u4 = deconv2d(u3, d3, self.gf*4)
        u5 = deconv2d(u4, d2, self.gf*2)
        u6 = deconv2d(u5, d1, self.gf)

        u7 = UpSampling2D(size=2)(u6)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)

        return Model([d01, d02], output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)
        img_C = Input(shape=self.img_shape)
        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate()([img_A, img_B, img_C])
        # print(combined_imgs.shape)

        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model([img_A, img_B, img_C], validity)

    def train(self, epochs, batch_size=10, sample_interval=50):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)
        file1 = os.listdir('/home/samik/Documents/P/405_ann/')
        file2 = os.listdir('/home/samik/Documents/P/405_imgs/')
        file3 = os.listdir('/home/samik/Documents/P/405_simp_v2/')
        # self.generator.load_weights('Models/G/generator1_134_400')
        # self.discriminator.load_weights('Models/D/discriminator1_134_400')
        # file1 = os.listdir('/home/samik/mnt/gpu3/mnt/disk128/main/training_data/training_data/STP/P/405_ann/')
        # file2 = os.listdir('/home/samik/mnt/gpu3/mnt/disk128/main/training_data/training_data/STP/P/405_imgs/')
        # file3 = os.listdir('/home/samik/mnt/gpu3/mnt/disk128/main/training_data/training_data/STP/P/405_simp_v2/')
        n_batches = 1000
        for epoch in range(0,epochs):
            for batch_i in range(n_batches):
                [imgs_A, imgs_B, imgs_C] = imageLoaderReturn(file1, batch_i*batch_size, batch_size)
                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Condition on B and generate a translated version
                fake_A = self.generator.predict([imgs_B, imgs_C])

                # Train the discriminators (original images = real / generated = Fake)
                d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B, imgs_C], valid)
                d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B, imgs_C], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # -----------------
                #  Train Generator
                # -----------------

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B, imgs_C], [valid, imgs_A])

                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] time: %s" % (epoch, epochs,
                                                                        batch_i, len(file1),
                                                                        d_loss,
                                                                        g_loss[0],
                                                                        elapsed_time))

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, np.random.randint(1000,1770))
                    self.generator.save_weights('Models/G/generator2_%d' % epoch, True)
                    self.discriminator.save_weights('Models/D/discriminator2_%d' % epoch,True)
                    # self.combined.save_weights()

    def sample_images(self, epoch, batch_i):
        # os.makedirs('Images/Output/')
        r, c = 4, 4
        file1 = os.listdir('/home/samik/Documents/P/405_ann/')
        file2 = os.listdir('/home/samik/Documents/P/405_imgs/')
        file3 = os.listdir('/home/samik/Documents/P/405_simp_v2/')

        # file1 = os.listdir('/home/samik/mnt/gpu3/mnt/disk128/main/training_data/training_data/STP/P/405_ann/')
        # file2 = os.listdir('/home/samik/mnt/gpu3/mnt/disk128/main/training_data/training_data/STP/P/405_imgs/')
        # file3 = os.listdir('/home/samik/mnt/gpu3/mnt/disk128/main/training_data/training_data/STP/P/405_simp_v2/')
        imgs_A, imgs_B, imgs_C = imageLoaderReturn(file1, batch_i, batch_size=4)
        fake_A = self.generator.predict([imgs_B, imgs_C])

        gen_imgs = np.concatenate([imgs_B, imgs_C, fake_A, imgs_A])

        # Rescale images 0 - 1
        gen_imgs = 127.5 * gen_imgs + 127.5

        # print(gen_imgs[3,].shape)
        # print(np.unique(cv2.threshold(np.squeeze(gen_imgs[8,]),0.,255.,cv2.THRESH_BINARY)[1]))
        titles = ['Input', 'Morse', 'Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        # plt.show()
        cnt = 0
        for i in range(r):
            for j in range(c):
                if cnt < r:
                    axs[i, j].imshow(np.squeeze(gen_imgs[cnt,]))
                axs[i,j].imshow(cv2.threshold(np.squeeze(gen_imgs[cnt,]),0.,255.,cv2.THRESH_BINARY)[1])
                axs[i, j].set_title(titles[i])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("Output3/%d_%d.png" % (epoch, batch_i))
        plt.close()




if __name__ == '__main__':
    gan = Pix2Pix()
    gan.train(epochs=2000, batch_size=1, sample_interval=200)
