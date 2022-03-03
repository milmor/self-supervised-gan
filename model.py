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
        super(GLU, self).__init__(**kwargs)

    def call(self, x):
        self.nc = x.shape[-1] // 2
        if len(x.shape) == 2:
            return x[:, :self.nc] * tf.nn.sigmoid(x[:, self.nc:])

        elif len(x.shape) == 4:
            return x[:, :, :, :self.nc] * tf.nn.sigmoid(x[:, :, :, self.nc:])


class NoiseInjection(layers.Layer):
    def __init__(self, **kwargs):
        super(NoiseInjection, self).__init__(**kwargs)
        self.weight = self.add_weight(
            'weight', shape=(1), initializer='zeros', 
            dtype=self.dtype, trainable=True)

    def call(self, feat):
        batch, height, width, _ = feat.shape
        noise = tf.random.normal((batch, height, width, 1), dtype=self.dtype)
        return feat + self.weight * noise


class UpBlockComp(layers.Layer):
    def __init__(self, filters, kernel_size=3, initializer='orthogonal'):
        super(UpBlockComp, self).__init__()
        self.up_conv = tf.keras.Sequential([
            layers.UpSampling2D(2),
            layers.Conv2D(filters*2, kernel_size=kernel_size, 
                padding='same', use_bias=False, kernel_initializer=initializer),
        ])
        self.noise_1 = NoiseInjection()
        self.act_1 = tf.keras.Sequential([
            layers.BatchNormalization(),
            GLU()
        ])
        self.conv = layers.Conv2D(filters*2, kernel_size=kernel_size, 
            padding='same', use_bias=False, kernel_initializer=initializer)
        self.noise_2 = NoiseInjection()
        self.act_2 = tf.keras.Sequential([
            layers.BatchNormalization(),
            GLU()
        ])

    def call(self, x):
        x = self.up_conv(x)
        x = self.noise_1(x)
        x = self.act_1(x)
        x = self.conv(x)
        x = self.noise_2(x)
        x = self.act_2(x)
        return x
    

def upBlock(filters, kernel_size=3, initializer='orthogonal'):
    block = tf.keras.Sequential([
            layers.UpSampling2D(2),
            layers.Conv2D(filters*2, kernel_size=kernel_size, 
                padding='same', use_bias=False, kernel_initializer=initializer),
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


class Generator(tf.keras.models.Model):
    def __init__(self, filters=1024, initializer='orthogonal'):
        super(Generator, self).__init__()
        self.init = InitLayer()
        
        self.up_8 = UpBlockComp(filters, initializer=initializer)
        self.up_16 = upBlock(filters // 2, initializer=initializer)
        self.up_32 = UpBlockComp(filters // 4, initializer=initializer)
       
        self.up_64 = upBlock(filters // 8, initializer=initializer)
        self.se_64 = SEBlock(filters // 8, initializer=initializer)
        
        self.up_128 = UpBlockComp(filters // 16, initializer=initializer)
        self.se_128 = SEBlock(filters // 16, initializer=initializer)
        
        self.up_256 = upBlock(filters // 32, initializer=initializer)
        self.se_256 = SEBlock(filters // 32, initializer=initializer)
        self.ch_conv = layers.Conv2D(3, 3, padding='same', kernel_initializer=initializer)
        self.tanh = layers.Activation('tanh', dtype='float32')
                       
    def call(self, z):
        z = normalize_2nd_moment(z)
        feat_4 = self.init(z)
        feat_8 = self.up_8(feat_4)   
        feat_16 = self.up_16(feat_8) 
        feat_32 = self.up_32(feat_16) 
        feat_64 = self.up_64(feat_32) * self.se_64(feat_4)
        feat_128 = self.up_128(feat_64) * self.se_128(feat_8)
        feat_256 = self.up_256(feat_128) * self.se_256(feat_16)
        img_256 = self.ch_conv(feat_256)
        return [self.tanh(img_256)]


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


def decode_image(filters=128, initializer='orthogonal'):
    encode_img = tf.keras.Sequential([
        upBlock(filters, initializer=initializer),
        upBlock(filters // 2, initializer=initializer),
        upBlock(filters // 4, initializer=initializer),
        upBlock(filters // 8, initializer=initializer),
        layers.Conv2D(3, 3, padding='same', kernel_initializer=initializer),
        layers.Activation('tanh', dtype='float32')
    ])
    return encode_img


class Discriminator(tf.keras.models.Model):
    def __init__(self, filters=128, initializer='orthogonal',
                 dec_dim=128):
        super(Discriminator, self).__init__()
        '''Encode image'''
        self.down_from_big = tf.keras.Sequential([
            layers.Conv2D(filters // 32, kernel_size=3, padding='same', 
                use_bias=False, kernel_initializer=initializer),
            layers.LeakyReLU(0.2)
        ])
        self.down_128 = DownBlockComp(filters // 16, initializer=initializer)
        self.down_64 = DownBlockComp(filters // 8, initializer=initializer)
        self.down_32 = DownBlockComp(filters // 4, initializer=initializer)
        self.down_16 = DownBlockComp(filters // 2, initializer=initializer)
        self.down_8 = DownBlockComp(filters, initializer=initializer)

        '''Logits'''
        self.logits = tf.keras.Sequential([
            layers.Conv2D(filters, kernel_size=1, padding='valid', 
                use_bias=False, kernel_initializer=initializer),  
            layers.LeakyReLU(0.2),
            layers.Conv2D(1, kernel_size=4, padding='valid', 
                use_bias=False, kernel_initializer=initializer), 
            layers.Flatten()
        ])
        self.decoder = decode_image(dec_dim, initializer=initializer)
 
    def call(self, img, decode=False):
        feat_256 = self.down_from_big(img)
        feat_128 = self.down_128(feat_256)  
        feat_64 = self.down_64(feat_128)  
        feat_32 = self.down_32(feat_64)  
        feat_16 = self.down_16(feat_32)
        feat_8 = self.down_8(feat_16)

        if decode:
            return [self.logits(feat_8), self.decoder(feat_8)]
        else:
            return [self.logits(feat_8)]

