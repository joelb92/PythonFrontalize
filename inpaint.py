import numpy as np
import tensorflow as tf
import cv2
import tqdm
import os
import matplotlib.pyplot as plt
import sys
import skimage.transform
# sys.path.append('..')
from network import Network

class Inpaint:
    IMAGE_SIZEx = 128
    IMAGE_SIZEy = 128
    LOCAL_SIZE = 64
    HOLE_MIN = 24
    HOLE_MAX = 48
    BATCH_SIZE = 1
    PRETRAIN_EPOCH = 100

    def __init__(self,modelLocation):
        self.x = tf.placeholder(tf.float32, [self.BATCH_SIZE, self.IMAGE_SIZEy, self.IMAGE_SIZEx, 3])
        self.mask = tf.placeholder(tf.float32, [self.BATCH_SIZE, self.IMAGE_SIZEy, self.IMAGE_SIZEx, 1])
        self.local_x = tf.placeholder(tf.float32, [self.BATCH_SIZE, self.LOCAL_SIZE, self.LOCAL_SIZE, 3])
        self.global_completion = tf.placeholder(tf.float32, [self.BATCH_SIZE, self.IMAGE_SIZEy, self.IMAGE_SIZEx, 3])
        self.local_completion = tf.placeholder(tf.float32, [self.BATCH_SIZE, self.LOCAL_SIZE, self.LOCAL_SIZE, 3])
        self.is_training = tf.placeholder(tf.bool, [])
        self.model = Network(self.x, self.mask, self.local_x, self.global_completion, self.local_completion, self.is_training, batch_size=self.BATCH_SIZE)
        self.sess = tf.Session()
        self.init_op = tf.global_variables_initializer()
        self.sess.run(self.init_op)
        saver = tf.train.Saver()
        saver.restore(self.sess, modelLocation)

    def inpaint(self, image, mask):
        originalSize = image.shape
        rimg = skimage.transform.resize(image, (self.IMAGE_SIZEy, self.IMAGE_SIZEx), preserve_range=True, mode='constant')
        # rimg = cv2.resize(image, (self.IMAGE_SIZEy, self.IMAGE_SIZEx))
        rimg = rimg / 127.5 - 1
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # rimg_normed = rimg
        # rmask = cv2.resize(mask, (self.IMAGE_SIZEy, self.IMAGE_SIZEx))
        # rmask = np.reshape(rmask, (128, 128, 1))
        rmask = skimage.transform.resize(mask, (self.IMAGE_SIZEy, self.IMAGE_SIZEx), preserve_range=True, mode='constant')
        rmask = np.reshape(rmask, (self.IMAGE_SIZEy, self.IMAGE_SIZEx, 1))
        rmask[rmask < .9] = 0
        rmask[rmask >= .9] = 1
        completion = self.sess.run(self.model.completion, feed_dict={self.x: np.array([rimg]),
                                                                     self.mask: np.array([rmask], dtype=np.uint8),
                                                                     self.is_training: False})

        img = completion[0]
        raw = np.array((rimg + 1) * 127.5,dtype=np.uint8) #raw
        # chmask = np.dstack((rmask, rmask, rmask))
        # goodPx = np.asarray(img * chmask, dtype=np.uint8)
        # outImg = np.asarray(rimg * (1 - chmask) + goodPx, dtype=np.uint8)
        # outImg = cv2.resize(outImg, image.shape[:2]) #img
        outputImage = np.array((img + 1) * 127.5, dtype=np.uint8)
        # masked = raw * (1 - mask) + np.ones_like(raw) * mask * 255
        return outputImage