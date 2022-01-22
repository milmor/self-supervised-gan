# Self-Supervised GAN
Implementation of the paper:

> Bingchen Liu, Yizhe Zhu, Kunpeng Song and Ahmed Elgammal. [Towards Faster and Stabilized GAN Training for High-fidelity Few-shot Image Synthesis](https://arxiv.org/abs/2101.04775). 

![Architecture](./images/disc_arch.png)

See [here](https://github.com/odegeasslbc/FastGAN-pytorch) for the official Pytorch implementation.


## Examples
![](images/animation_1.gif "img 1")
![](images/animation_2.gif "img 2")
![](images/animation_3.gif "img 3")


## Dependencies
- Python 3.8
- Tensorfow 2.7


## Usage
### Train
Use `--train_dir=<train_dataset_path>` to provide the dataset path. 
```
python train.py --train_dir=<train_dataset_path> -
```

## Licence
MIT
