 ---

<div align="center">    
 
# Robust Test-Time Adaptation in Dynamic Scenarios (CVPR 2023)

[Longhui Yuan](https://yuanlonghui.github.io), [Binhui Xie](https://binhuixie.github.io), [Shuang Li](https://shuangli.xyz)


[![Paper](https://img.shields.io/badge/Paper-arXiv-%23B31B1B?style=flat-square)](https://arxiv.org/abs/2303.13899)&nbsp;&nbsp;

</div>

<!-- - [Overview](#overview)
- [Prerequisites Installation](#prerequisites)
- [Datasets Preparation](#datasets-preparation)
- [Code Running](#code-running)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact) -->
<!-- - [Citation](#citation) -->



## Overview
We propose a new test-time adaptation setup that is more suitable for real-world applications, namely practical test-time adaptation (PTTA). 
PTTA considers both distribution changing and correlation sampling.
Meanwhile, we propose a robust test-time adaptation (RoTTA) method, which has a more comprehensive consideration of the challenges of PTTA.

![image](./resources/framework.png)

## Prerequisites
Step by Step installation,
```bash
conda create -n rotta python=3.9.0
conda activate rotta

# this installs the right pip and dependencies for the fresh python
conda install -y ipython pip

# this installs required packages
pip install -r requirements.txt
```


## Datasets Preparation
Download [CIFAR-10-C](https://zenodo.org/record/2535967#.ZDETTHZBxhF) and [CIFAR-100-C](https://zenodo.org/record/3555552#.ZDES-XZBxhE). (Running the code directly also works, since it automatically downloads the data set at the first running, but it's too slow to tolerate and has high requirements on internet stability)

Symlink dataset by
```bash
ln -s path_to_cifar10_c datasets/CIFAR-10-C
ln -s path_to_cifar100_c datasets/CIFAR-100-C
```

## Code Running
Run RoTTA by
```bash
python ptta.py \
      -acfg configs/adapter/rotta.yaml \
      -dcfg configs/dataset/cifar10.yaml \
      OUTPUT_DIR RoTTA/cifar10

python ptta.py \
      -acfg configs/adapter/rotta.yaml \
      -dcfg configs/dataset/cifar100.yaml \
      OUTPUT_DIR RoTTA/cifar100
```

## Acknowledgements
This project is based on the following open-source projects: 
- [tent](https://github.com/DequanWang/tent) 
- [cotta](https://github.com/qinenergy/cotta)

We thank their authors for making the source code publicly available.


## Citation

If you find this project useful in your research, please consider citing:
```bibtex
@inproceedings{yuan2023robust,
  title={Robust test-time adaptation in dynamic scenarios},
  author={Yuan, Longhui and Xie, Binhui and Li, Shuang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={15922--15932},
  year={2023}
}
``` 

## Contact
If you have any problem with our code, feel free to contact 
- longhuiyuan@bit.edu.cn

or describe your problem in Issues.



