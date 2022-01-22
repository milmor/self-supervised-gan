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
from train import *


def run_training(args):
    train_dir = args.train_dir
    main_dir = args.main_dir
    run_dir = args.run_dir

    ckpt_interval = args.ckpt_interval
    epochs = args.epochs
    train_seed = args.train_seed
    test_seed = args.test_seed
    max_ckpt_to_keep = args.max_ckpt_to_keep

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

    log_dir = os.path.join(model_dir, 'log-dir')
    writer = tf.summary.create_file_writer(log_dir)

    gen_test_dir = os.path.join(model_dir, 'test-gen')
    disc_test_dir = os.path.join(model_dir, 'test-rec')

    os.makedirs(gen_test_dir, exist_ok=True)
    os.makedirs(disc_test_dir, exist_ok=True)

    train_ds = create_train_ds(train_dir, hparams['batch_size'], train_seed)

    generator = Generator_64(filters=hparams['g_dim'], 
                             initializer=hparams['g_initializer'])
    discriminator = Discriminator(filters=hparams['d_dim'], 
                                  initializer=hparams['d_initializer'], 
                                  dec_dim=hparams['dec_dim'])

    g_loss, d_loss = get_loss(hparams['loss'])
    perc_loss = LossNetwork(128, hparams['rec_layers'])

    g_opt = tf.keras.optimizers.Adam(learning_rate=hparams['g_learning_rate'], 
                                     beta_1=hparams['g_beta_1'], 
                                     beta_2=hparams['g_beta_2'])

    d_opt = tf.keras.optimizers.Adam(learning_rate=hparams['d_learning_rate'], 
                                     beta_1=hparams['d_beta_1'],
                                     beta_2=hparams['d_beta_2'])

    num_examples_to_generate = 64
    noise_seed = tf.random.normal([num_examples_to_generate, 
                                   hparams['noise_dim']], seed=test_seed)
    
    # Metrics
    gen_loss_avg = tf.keras.metrics.Mean()
    disc_loss_avg = tf.keras.metrics.Mean()
    gp_avg = tf.keras.metrics.Mean()
    rec_avg = tf.keras.metrics.Mean()
    disc_total_loss_avg = tf.keras.metrics.Mean()

    metrics = [gen_loss_avg, disc_loss_avg,
               gp_avg, rec_avg, disc_total_loss_avg]

    checkpoint_dir = os.path.join(model_dir, 'training-checkpoints')
    ckpt = tf.train.Checkpoint(generator_optimizer=g_opt,
                               discriminator_optimizer=d_opt,
                               generator=generator,
                               discriminator=discriminator,
                               epoch=tf.Variable(0))
    ckpt_manager = tf.train.CheckpointManager(ckpt, directory=checkpoint_dir, 
                                              max_to_keep=max_ckpt_to_keep)

    ckpt.restore(ckpt_manager.latest_checkpoint)

    train_batch = next(iter(train_ds))

    for _ in range(int(ckpt.epoch), epochs):
        start = time.time()
        step_int = int(ckpt.epoch)
        
        # Clear metrics
        gen_loss_avg.reset_states()
        disc_loss_avg.reset_states()
        rec_avg.reset_states()
        gp_avg.reset_states()

        for image_batch in train_ds:
            train_step(image_batch, generator, discriminator, g_opt, d_opt, 
               g_loss, d_loss, perc_loss, metrics)

        # Print and save Tensorboard
        print('Time for epoch {} is {} sec'.format(step_int, time.time()-start))
        print('Generator loss: {:.4f}'.format(gen_loss_avg.result()))
        print('Discriminator loss: {:.4f}'.format(disc_loss_avg.result()))
        print('GP: {:.4f}'.format(gp_avg.result())) 
        print('Reconstruction loss: {:.4f}'.format(rec_avg.result())) 
        print('Discriminator total loss: {:.4f}'.format(disc_total_loss_avg.result()))    
    
        with writer.as_default():
            tf.summary.scalar('generator_loss', gen_loss_avg.result(), step=step_int)
            tf.summary.scalar('discriminator_loss', disc_loss_avg.result(), step=step_int)
            tf.summary.scalar('gp_loss', gp_avg.result(), step=step_int)
            tf.summary.scalar('reconstruction_loss', rec_avg.result(), step=step_int)

        # Generate and save test images plot
        save_generator_img(generator, step_int, noise_seed, gen_test_dir)
        save_decoder_img(discriminator, step_int, train_batch, disc_test_dir)

        if (step_int) % ckpt_interval == 0:
            ckpt_manager.save(step_int)
      
        ckpt.epoch.assign_add(1)


@tf.function
def train_step(real_images, generator, discriminator, g_opt, d_opt, 
               gen_loss, disc_loss, rec_loss, metrics):
    gen_loss_avg, disc_loss_avg, disc_total_loss_avg, gp_avg, rec_avg = metrics
    
    noise = tf.random.normal([hparams['batch_size'], hparams['noise_dim']])
    
    # Train the discriminator
    for i in range(hparams['d_steps']):
        with tf.GradientTape() as disc_tape:
            generator_output = generator(noise, training=True)
            real_images = DiffAugment(real_images, hparams['policy'])
            fake_images = DiffAugment(generator_output[0], hparams['policy'])
            
            real_disc_output = discriminator(real_images, decode=True, training=True)
            fake_disc_output = discriminator(fake_images, training=True)

            d_loss = disc_loss(real_disc_output[0], fake_disc_output[0])
            r_loss = rec_loss(real_images, real_disc_output[1]) * hparams['rec_weight']
            
            gp = 0.0
            if hparams['gp_weight'] != 0:
                gp = gradient_penalty(
                    discriminator, real_images, fake_images) * hparams['gp_weight']
                
            d_total = d_loss + gp + r_loss

        disc_gradients = disc_tape.gradient(d_total, discriminator.trainable_variables)
        d_opt.apply_gradients(zip(disc_gradients, discriminator.trainable_variables)) 
        disc_loss_avg(d_loss)  
        gp_avg(gp)
        rec_avg(r_loss)
        disc_total_loss_avg(d_total)
        
    noise = tf.random.normal([hparams['batch_size'], hparams['noise_dim']])
    
    # Train the generator
    with tf.GradientTape() as gen_tape:
        generator_output = generator(noise, training=True)
        fake_images = DiffAugment(generator_output[0], hparams['policy'])
        fake_disc_output = discriminator(fake_images, training=True)
        
        gen_loss = gen_loss(fake_disc_output[0])
        
    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    g_opt.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    gen_loss_avg(gen_loss)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir')
    parser.add_argument('--main_dir', default='model-1')
    parser.add_argument('--run_dir', default='run-1')
    parser.add_argument('--ckpt_interval', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--train_seed', type=int, default=15)    
    parser.add_argument('--test_seed', type=int, default=15)  
    parser.add_argument('--max_ckpt_to_keep', type=int, default=5)  
    args = parser.parse_args()

    run_training(args)


if __name__ == '__main__':
    main()
