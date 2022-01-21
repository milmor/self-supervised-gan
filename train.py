'''
Author: Emilio Morales (mil.mor.mor@gmail.com)
        Jan 2022
'''
import tensorflow as tf
from diffaug import DiffAugment
from hparams import hparams
from utils import *


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
                    discriminator, real_images, 
                    generator_output[0]) * hparams['gp_weight']
                
            d_total = d_loss + gp + r_loss

        disc_gradients = disc_tape.gradient(d_total, discriminator.trainable_variables)
        d_opt.apply_gradients(zip(disc_gradients, discriminator.trainable_variables)) 
        disc_loss_avg(d_loss)
        rec_avg(r_loss)
        disc_total_loss_avg(d_total)
        gp_avg(gp)
        
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
