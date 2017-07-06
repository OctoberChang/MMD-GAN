# MMD-GAN
Code accompanying the paper [MMD-GAN: Towards Deeper Understanding of Moment Matching Network](https://arxiv.org/abs/1705.08584).

# Prerequisites
    - Python, NumPy, Scipy
    - PyTorch (v0.1.12)
    - A recent Nvidia GPU

# Usage
```
./mmd_gan [OPTIONS]
OPTIONS:
    --dataset DATASET: type of dataset (mnist/cifar10/celeba/lsun)
    --dataroot DATAROOT: path to dataset
    --workers WORKERS: number of threads to load data
    --batch_size BATCH_SIZE: batch size for training
    --image_size IMAGE_SIZE: image size of dataset
    --nc NC: number of channels in images
    --nz NZ: hidden dimension in z and codespace
    --max_iter MAX_ITER: max iteration for training
    --lr LR: learning rate (default 5e-5)
    --gpu_device GPU_DEVICE: gpu id (default 0)
    --netG NETG: path to generator model
    --netD NETD: path to discriminator model
    --Diters DITERS: number of updates for discriminator per one generator update
    --experiment EXPERIMENT: output directory of sampled images
```

For a quick start, please set the DATA_PATH variable in run_exp.sh to
```
    ./data
```
and run
```
	$ ./run_exp.sh [mnist/cifar10/celeba/lsun]
```

# Dataset
For mnist and cifar10, the dataset will be automatically download if not exist in
the designated DATAROOT directory.

For CelebA and LSUN dataset, please run the download script in ./data directory.


# More Info
This repository is by
[Chun-Liang Li](http://www.cs.cmu.edu/~chunlial/),
[Wei-Cheng Chang](https://octoberchang.github.io/),
[Yu Cheng](https://sites.google.com/site/chengyu05/),
[Yiming Yang](http://www.cs.cmu.edu/~yiming/),
[Barnabás Póczos](http://www.cs.cmu.edu/~bapoczos/),
and contains the source code to
reproduce the experiments in our paper
[MMD GAN: Towards Deeper Understanding of Moment Matching Network](https://arxiv.org/abs/1705.08584).
If you find this repository helpful in your publications, please consider citing our paper.
```
@article{li2017mmd,
    title={MMD GAN: Towards Deeper Understanding of Moment Matching Network},
    author={Li, Chun-Liang and Chang, Wei-Cheng and Cheng, Yu and Yang, Yiming and P{\'o}czos, Barnab{\'a}s},
    journal={arXiv preprint arXiv:1705.08584},
    year={2017}
}
```

For any questions and comments, please send your email to
[wchang2@cs.cmu.edu](mailto:wchang2@cs.cmu.edu)

