'''FastGAN model for Tensorflow.

Reference:
  - Bingchen Liu, Yizhe Zhu, Kunpeng Song and Ahmed Elgammal. 
    [Towards Faster and Stabilized GAN Training for High-fidelity Few-shot Image Synthesis](
      https://arxiv.org/abs/2101.04775) 
  
Author: Emilio Morales (mil.mor.mor@gmail.com)
        Jan 2022
'''
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow_addons.layers import AdaptiveAveragePooling2D


def normalize_2nd_moment(x, axis=1, eps=1e-8):
    return x * tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=axis, keepdims=True) + eps)


class GLU(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        self.nc = x.shape[-1] // 2
        if len(x.shape) == 2:
            return x[:, :self.nc] * tf.nn.sigmoid(x[:, self.nc:])

        elif len(x.shape) == 4:
            return x[:, :, :, :self.nc] * tf.nn.sigmoid(x[:, :, :, self.nc:])


class NoiseInjection(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.weight = self.add_weight(
            self.name, shape=(), initializer="zeros", trainable=True)

    def call(self, feat):
        batch, height, width, _ = feat.shape
        noise = tf.random.normal((batch, height, width, 1), dtype=self.dtype)
        return feat + self.weight * noise


def upBlockComp(filters, kernel_size=3, initializer='orthogonal'):
    block = tf.keras.Sequential([
            layers.UpSampling2D(2),
            layers.Conv2D(filters*2, kernel_size=kernel_size, 
                padding="same", use_bias=False, kernel_initializer=initializer),
            NoiseInjection(),
            layers.BatchNormalization(),
            GLU(),
            layers.Conv2D(filters*2, kernel_size=kernel_size, 
                padding="same", use_bias=False, kernel_initializer=initializer),
            NoiseInjection(),
            layers.BatchNormalization(),
            GLU()
    ])
    return block


def upBlock(filters, kernel_size=3, initializer='orthogonal'):
    block = tf.keras.Sequential([
            layers.UpSampling2D(2),
            layers.Conv2D(filters*2, kernel_size=kernel_size, 
                padding="same", use_bias=False, kernel_initializer=initializer),
            layers.BatchNormalization(),
            GLU()
    ])
    return block


def SEBlock(filters, kernel_size=4, activation='swish', initializer='orthogonal'):
    block = tf.keras.Sequential([
                AdaptiveAveragePooling2D((4, 4)),
                layers.Conv2D(filters=filters, 
                    kernel_size=4, 
                    activation=activation, 
                    use_bias=False, kernel_initializer=initializer),
                layers.Conv2D(filters=filters, 
                    kernel_size=1, 
                    activation='sigmoid', 
                    use_bias=False, kernel_initializer=initializer)
    ])
    return block


class InitLayer(layers.Layer):
    def __init__(self, units=256, initializer='orthogonal'):
        super(InitLayer, self).__init__()
        self.conv = tf.keras.Sequential([
            layers.Conv2DTranspose(units * 2, 
                kernel_size=4, 
                use_bias=False, kernel_initializer=initializer),
                layers.BatchNormalization(),
                GLU()
        ])
        
    def call(self, x):
        x = tf.expand_dims(x, 1)
        x = tf.expand_dims(x, 1)
        return self.conv(x)


class DownBlockComp(layers.Layer):
    def __init__(self, filters, initializer='orthogonal'):
        super(DownBlockComp, self).__init__()

        self.main = tf.keras.Sequential([
            layers.Conv2D(filters, kernel_size=4, padding='same',
                strides=2, use_bias=False, kernel_initializer=initializer),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Conv2D(filters, kernel_size=3, padding='same',
                use_bias=False, kernel_initializer=initializer),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
        ])
        
        self.direct = tf.keras.Sequential([
            layers.AveragePooling2D((2, 2)),
            layers.Conv2D(filters, kernel_size=1, padding='same',
                use_bias=False, kernel_initializer=initializer),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
        ])

    def call(self, x):
        return (self.main(x) + self.direct(x)) / 2
