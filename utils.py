import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import vgg16

AUTOTUNE = tf.data.experimental.AUTOTUNE


def deprocess(img):
    return img * 127.5 + 127.5

def train_convert(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [256, 256])
    img = tf.image.random_flip_left_right(img)
    img = (img - 127.5) / 127.5 
    return img

def create_train_ds(train_dir, batch_size, seed=15):
    img_paths = tf.data.Dataset.list_files(str(train_dir))
    BUFFER_SIZE = tf.data.experimental.cardinality(img_paths)
     
    img_paths = img_paths.cache().shuffle(BUFFER_SIZE)
    ds = img_paths.map(train_convert, num_parallel_calls=AUTOTUNE).batch(
        batch_size, drop_remainder=True, num_parallel_calls=AUTOTUNE).prefetch(
        AUTOTUNE)
    print(f'Train dataset size: {BUFFER_SIZE}')
    print(f'Train batches: {tf.data.experimental.cardinality(ds)}')
    return ds

def save_generator_img(model, epoch, noise, direct):
    predictions = model(noise, training=False)
    predictions = np.clip(deprocess(predictions[0]), 0, 255).astype(np.uint8)

    fig = plt.figure(figsize=(15, 15))

    for i in range(predictions.shape[0]):
        # create subplot and append to ax
        fig.add_subplot(8, 8, i+1)
        plt.imshow(predictions[i, :, :, :])
        plt.axis('off')

    path = os.path.join(direct, f'{epoch:04d}.png')
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.savefig(path, format='png')
    plt.close()

def save_decoder_img(model, epoch, img, direct):
    predictions = model(img, decode=True)
    predictions = np.clip(deprocess(predictions[1]), 0, 255).astype(np.uint8) 

    fig = plt.figure(figsize=(8, 4))

    for i in range(predictions.shape[0]):
        fig.add_subplot(2, 4, i+1)
        plt.imshow(predictions[i, :, :, :])
        plt.axis('off')

    path = os.path.join(direct, f'{epoch:04d}.png')
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.savefig(path, format='png')
    plt.close()

def get_loss(loss):
    if loss == 'bce':
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        def discriminator_loss(real_img, fake_img):
            real_loss = cross_entropy(tf.ones_like(real_img), real_img)
            fake_loss = cross_entropy(tf.zeros_like(fake_img), fake_img)
            return real_loss + fake_loss

        def generator_loss(fake_img):
            return cross_entropy(tf.ones_like(fake_img), fake_img)

        return generator_loss, discriminator_loss

    elif loss == 'hinge':
        def d_real_loss(logits):
            return tf.reduce_mean(tf.nn.relu(1.0 - logits))

        def d_fake_loss(logits):
            return tf.reduce_mean(tf.nn.relu(1.0 + logits))

        def discriminator_loss(real_img, fake_img):
            real_loss = d_real_loss(real_img)
            fake_loss = d_fake_loss(fake_img)
            return fake_loss + real_loss

        def generator_loss(fake_img):
            return -tf.reduce_mean(fake_img)

        return generator_loss, discriminator_loss

    elif loss == 'wgan':
        def discriminator_loss(real_img, fake_img):
            real_loss = tf.reduce_mean(real_img)
            fake_loss = tf.reduce_mean(fake_img)
            return fake_loss - real_loss 

        def generator_loss(fake_img):
            return -tf.reduce_mean(fake_img)

        return generator_loss, discriminator_loss


class LossNetwork(tf.keras.models.Model):
    def __init__(self, input_size=128, 
                 content_layers = ['block1_conv2', 
                                   'block2_conv2', 
                                   'block3_conv3'],
        ):
        super(LossNetwork, self).__init__()
        self.res = layers.experimental.preprocessing.Resizing(input_size, input_size)
        
        vgg = vgg16.VGG16(include_top=False, weights='imagenet')
        vgg.trainable = False
        model_outputs = [vgg.get_layer(name).output for name in content_layers]
        self.model = tf.keras.models.Model(vgg.input, model_outputs)
        self.linear = layers.Activation('linear', dtype='float32') 

    def call(self, real_img, rec_img):
        real_img = deprocess(real_img)
        real_img = self.res(real_img)
        real_img = vgg16.preprocess_input(real_img)
        real_maps = self.model(real_img)
    
        rec_img = deprocess(rec_img)
        rec_img = self.res(rec_img)
        rec_img = vgg16.preprocess_input(rec_img)
        rec_maps = self.model(rec_img)
        
        loss = tf.add_n([tf.reduce_mean(tf.keras.losses.MAE(real, rec)) 
                    for real, rec in zip(real_maps, rec_maps)])
        return loss
