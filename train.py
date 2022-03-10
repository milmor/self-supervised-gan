'''
Author: Emilio Morales (mil.mor.mor@gmail.com)
        Jan 2022
'''
import argparse
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable tensorflow debugging logs
import time
import tensorflow as tf
import json
from diffaug import DiffAugment
from model import *
from utils import *


class FastGAN(tf.keras.Model):
    def __init__(self, generator, discriminator, noise_dim, gp_weight, rec_weight, policy, d_steps):
        super(FastGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.noise_dim = noise_dim
        self.gp_weight = gp_weight
        self.rec_weight = rec_weight
        self.policy = policy
        self.d_steps = d_steps
        
        # Metrics
        self.g_loss_avg = tf.keras.metrics.Mean()
        self.d_loss_avg = tf.keras.metrics.Mean()
        self.gp_avg = tf.keras.metrics.Mean()
        self.rec_avg = tf.keras.metrics.Mean()
        self.d_total_avg = tf.keras.metrics.Mean()

    def compile(self, g_optimizer, d_optimizer, g_loss, d_loss, rec_loss):
        super(FastGAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.g_loss = g_loss
        self.d_loss = d_loss
        self.rec_loss = rec_loss
        
    def gradient_penalty(self, real_samples, fake_samples):
        alpha = tf.random.uniform([real_samples.shape[0], 1, 1, 1], minval=0., maxval=1.)
        diff = fake_samples - real_samples
        interpolation = real_samples + alpha * diff

        with tf.GradientTape() as gradient_tape:
            gradient_tape.watch(interpolation)
            pred = self.discriminator(DiffAugment(interpolation, self.policy), training=True)

        gradients = gradient_tape.gradient(pred[0], [interpolation])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean((norm - 1.0) ** 2)
        return gradient_penalty

    @tf.function
    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal(shape=[batch_size, self.noise_dim])

        # Train the discriminator
        for _ in range(self.d_steps):
            with tf.GradientTape() as d_tape:
                generator_output = self.generator(noise, training=True)
                real_aug = DiffAugment(real_images, self.policy)
                fake_aug = DiffAugment(generator_output[0], self.policy)
                real_disc_output = self.discriminator(real_aug, decode=True, training=True)
                fake_disc_output = self.discriminator(fake_aug, training=True)
                
                d_loss = self.d_loss(real_disc_output[0], fake_disc_output[0])
                
                rec_loss = self.rec_loss(real_aug, real_disc_output[1]) * self.rec_weight
                
                gp = 0.0
                if self.gp_weight != 0:
                    gp = self.gradient_penalty(real_images, generator_output[0]) * self.gp_weight
                
                d_total = d_loss + rec_loss + gp 
                
            d_gradients = d_tape.gradient(d_total, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(
                zip(d_gradients, self.discriminator.trainable_weights)
            )
            # Save discriminator metrics
            self.d_loss_avg(d_loss)
            self.gp_avg(gp)
            self.rec_avg(rec_loss)
            self.d_total_avg(d_total)

        noise = tf.random.normal(shape=[batch_size, self.noise_dim])

        # Train the generator 
        with tf.GradientTape() as g_tape:
            generator_output = self.generator(noise, training=True)
            fake_aug = DiffAugment(generator_output[0], self.policy)
            fake_disc_output = self.discriminator(fake_aug, training=True)
            g_loss = self.g_loss(fake_disc_output[0])
            
        g_gradients = g_tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_weights))
        # Save generator metrics        
        self.g_loss_avg(g_loss)
        
    def create_log(self, model_dir, ckpt_interval, max_ckpt_to_keep):
        log_dir = os.path.join(model_dir, 'log-dir')
        self.writer = tf.summary.create_file_writer(log_dir)
        
        checkpoint_dir = os.path.join(model_dir, 'training-checkpoints')
        self.ckpt = tf.train.Checkpoint(g_optimizer=self.g_optimizer,
                                        d_optimizer=self.d_optimizer,
                                        generator=self.generator,
                                        discriminator=self.discriminator,
                                        epoch=tf.Variable(0))
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, directory=checkpoint_dir, 
                                                       max_to_keep=max_ckpt_to_keep)
        self.ckpt_interval = ckpt_interval
        
        if self.ckpt_manager.latest_checkpoint:    
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print('Checkpoint restored from {} at epoch {}'.format(self.ckpt_manager.latest_checkpoint,
                                                                   int(self.ckpt.epoch)))
            
    def save_log(self, verbose=1, reset_states=True):
        # Cast epoch
        epoch = int(self.ckpt.epoch)
        
        # Print metrics
        if verbose:
            print('Epoch: {}'.format(epoch))
            print('Generator loss: {:.4f}'.format(self.g_loss_avg.result()))
            print('Discriminator loss: {:.4f}'.format(self.d_loss_avg.result()))
            print('GP: {:.4f}'.format(self.gp_avg.result())) 
            print('Reconstruction loss: {:.4f}'.format(self.rec_avg.result())) 
            print('Discriminator total loss: {:.4f}\n'.format(self.d_total_avg.result()))      
            
        # Save metrics
        with self.writer.as_default():
            tf.summary.scalar('generator_loss', self.g_loss_avg.result(), step=epoch)
            tf.summary.scalar('discriminator_loss', self.d_loss_avg.result(), step=epoch)
            tf.summary.scalar('gp_loss', self.gp_avg.result(), step=epoch)
            tf.summary.scalar('reconstruction_loss', self.rec_avg.result(), step=epoch)
            tf.summary.scalar('discriminator_total_loss', self.d_total_avg.result(), step=epoch)
            
        # Reset metrics    
        if reset_states:
            self.g_loss_avg.reset_states()
            self.d_loss_avg.reset_states()
            self.gp_avg.reset_states()
            self.rec_avg.reset_states()
            self.d_total_avg.reset_states()
            
        # Save checlpoint    
        if epoch % self.ckpt_interval == 0:
            self.ckpt_manager.save(epoch)
            print('Checkpoint saved at epoch {}\n'.format(epoch)) 
            
        self.ckpt.epoch.assign_add(1)


def train(args):
    print('\n#########################')
    print('Self-Supervised GAN Train')
    print('#########################\n')
    file_pattern = args.file_pattern
    main_dir = args.main_dir
    run_dir = args.run_dir

    ckpt_interval = args.ckpt_interval
    epochs = args.epochs
    test_seed = args.test_seed
    max_ckpt_to_keep = args.max_ckpt_to_keep

    global hparams

    # Create directory
    model_dir = os.path.join(main_dir, run_dir)
    hparams_file = os.path.join(model_dir, run_dir + '_hparams.json')

    if os.path.isfile(hparams_file):
        with open(hparams_file) as f:
            hparams = json.load(f)
        print('hparams {} loaded'.format(hparams_file))
    else:
        from hparams import hparams
        os.makedirs(model_dir, exist_ok=True)
        json_hparams = json.dumps(hparams)
        with open(hparams_file, 'w') as f:
            f.write(json_hparams)
        print('hparams {} saved'.format(hparams_file))

    gen_test_dir = os.path.join(model_dir, 'test-gen')
    disc_test_dir = os.path.join(model_dir, 'test-rec')

    os.makedirs(gen_test_dir, exist_ok=True)
    os.makedirs(disc_test_dir, exist_ok=True)

    # Define model
    generator = Generator(filters=hparams['g_dim'], 
                             initializer=hparams['g_initializer'])
    discriminator = Discriminator(filters=hparams['d_dim'], 
                                  initializer=hparams['d_initializer'], 
                                  dec_dim=hparams['dec_dim'])

    gan = FastGAN(generator=generator, discriminator=discriminator, 
                  noise_dim=hparams['noise_dim'],
                  gp_weight=hparams['gp_weight'],
                  rec_weight=hparams['rec_weight'],
                  policy=hparams['policy'],
                  d_steps=hparams['d_steps'])

    # Create dataset and define losses
    train_ds = create_train_ds(file_pattern, hparams['batch_size'])
    generator_loss, discriminator_loss = get_loss(hparams['loss'])
    perc_loss = LossNetwork(128, hparams['rec_layers'])

    gan.compile(
        g_optimizer=tf.keras.optimizers.Adam(learning_rate=hparams['g_learning_rate'], 
                                             beta_1=hparams['g_beta_1'], 
                                             beta_2=hparams['g_beta_2']),
        d_optimizer=tf.keras.optimizers.Adam(learning_rate=hparams['d_learning_rate'], 
                                             beta_1=hparams['d_beta_1'],
                                             beta_2=hparams['d_beta_2']),
        g_loss=generator_loss,
        d_loss=discriminator_loss,
        rec_loss=perc_loss
    )

    gan.create_log(model_dir, ckpt_interval, max_ckpt_to_keep)

    # Log vars
    num_examples_to_generate = 64
    noise_seed = tf.random.normal([num_examples_to_generate, 
                                   hparams['noise_dim']], seed=test_seed)
    train_batch = next(iter(train_ds))
    gan.ckpt.epoch.assign_add(1)
    start_epoch = int(gan.ckpt.epoch)
    start_epoch

    for _ in range(start_epoch, epochs):
        start = time.time()
        for image_batch in train_ds:
            gan.train_step(image_batch)
            
        print('Time for epoch is {} sec'.format(time.time()-start))
        save_generator_img(gan.generator, int(gan.ckpt.epoch), noise_seed, gen_test_dir)
        save_decoder_img(gan.discriminator, int(gan.ckpt.epoch), train_batch, disc_test_dir)
        gan.save_log()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_pattern')
    parser.add_argument('--main_dir', default='model-1')
    parser.add_argument('--run_dir', default='run-1')
    parser.add_argument('--ckpt_interval', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=5000)  
    parser.add_argument('--test_seed', type=int, default=15)  
    parser.add_argument('--max_ckpt_to_keep', type=int, default=5)  
    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
