# Self-Supervised GAN
Implementation of the _FastGAN_ model in the paper:

> [Towards Faster and Stabilized GAN Training for High-fidelity Few-shot Image Synthesis](https://arxiv.org/abs/2101.04775). 

![Gen architecture](./images/gen_arch.png)
![Disc architecture](./images/disc_arch.png)

See [here](https://github.com/odegeasslbc/FastGAN-pytorch) for the official Pytorch implementation.


## Examples
![](images/animation_1.gif "img 1")


## Dependencies
- Python 3.8
- Tensorfow 2.8
- Tensorflow Addons 0.16


## Usage
### Train
Use `--file_pattern=<file_pattern>` to provide the dataset path and file pattern.
```
python train.py --file_pattern=./dataset_path/*.png
```

### Generate
Use `--main_dir=<main_dir>` to provide the model directory name.
```
python generate.py --main_dir=<main_dir>
```

### Hparams setting
Adjust hyperparameters on the `hparams.py` file.

### Tensorboard
Run `tensorboard --logdir ./`.


## References
Code:
- This model depends on other files that may be licensed under different open source licenses.
- Self-Supervised GAN uses [Differentiable Augmentation](https://arxiv.org/abs/2006.10738). Under BSD 2-Clause "Simplified" License.

Implementation notes:
- Self-supervised discriminator with a single reconstruction decoder and perceptual loss.
- Hinge loss GAN and WGAN gradient penalty.
- Skip-layer excitation generator.
- Orthogonal initialization.
- Adam with β1 = 0.5 and β2 = 0.99. 
- Batch size = 8.


## Licence
MIT
