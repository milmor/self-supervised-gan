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
from PIL import Image
from tqdm import tqdm
from model import Generator
from utils import *


def run_generate(args):
    print('\n#########################')
    print('Self-Supervised GAN Generate')
    print('#########################\n')
    main_dir = args.main_dir
    run_dir = args.run_dir
    n_images = args.n_images
    batch_size = args.batch_size
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

    # Define model
    generator = Generator(filters=hparams['g_dim'], 
                          initializer=hparams['g_initializer'])
    
    checkpoint_dir = os.path.join(model_dir, 'training-checkpoints')
    ckpt = tf.train.Checkpoint(generator=generator,
                               epoch=tf.Variable(0))
    ckpt_manager = tf.train.CheckpointManager(ckpt, directory=checkpoint_dir, 
                                              max_to_keep=1)

    if ckpt_manager.latest_checkpoint:    
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        print('\nCheckpoint restored from: {}'.format(ckpt_manager.latest_checkpoint))

    # Create epoch diractory
    test_dir = os.path.join(model_dir, 'test-dir')
    os.makedirs(test_dir, exist_ok=True)
    epoch_dir = os.path.join(test_dir, 'ep{}-{}'.format(str(ckpt.epoch.numpy()), str(n_images)))
    os.makedirs(epoch_dir, exist_ok=True)

    # Generate
    start = time.time()
    gen_img = gen_batches(generator, n_images, batch_size, hparams['noise_dim'], epoch_dir)
    print('Time: {:.4f} sec'.format(time.time()-start))  


def gen_batches(model, n_images, batch_size, noise_dim, epoch_dir):
    n_batches = n_images // batch_size
    n_used_imgs = n_batches * batch_size
    
    for i in tqdm(range(n_batches)):
        start = i * batch_size
        noise = tf.random.normal([batch_size, noise_dim])
        gen_batch = model(noise, training=False)
        gen_batch = np.clip(deprocess(gen_batch[0]), 0.0, 255.0)

        img_index = start
        for img in gen_batch:
            img = Image.fromarray(img.astype('uint8'))
            file_name = os.path.join(epoch_dir, '{}.jpg'.format(str(img_index)))
            img.save(file_name)
            img_index += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--main_dir', default='model-1')
    parser.add_argument('--run_dir', default='run-1')
    parser.add_argument('--n_images', type=int, default=500) 
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--test_seed', type=int, default=15)   
    args = parser.parse_args()

    run_generate(args)


if __name__ == '__main__':
    main()
