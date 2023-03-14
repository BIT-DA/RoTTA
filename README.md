## Robust Test-Time Adaptation in Dynamic Scenarios

PyTorch's implementation for RoTTA.

Firstly, create environment by
```bash
conda create -n rotta python=3.9.0
conda activate rotta
pip install -r requirements.txt
```

Link dataset by
```bash
ln -s path_to_cifar10_c datasets/CIFAR-10-C
ln -s path_to_cifar100_c datasets/CIFAR-100-C
```

Run RoTTA by
```bash
python ptta.py \
      -acfg configs/adapter/rotta.yaml \
      -dcfg configs/dataset/cifar10.yaml \
      OUTPUT_DIR roma/cifar10

python ptta.py \
      -acfg configs/adapter/rotta.yaml \
      -dcfg configs/dataset/cifar100.yaml \
      OUTPUT_DIR roma/cifar100
```
