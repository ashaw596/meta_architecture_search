# Meta Architecture Search

This repository contains the trained deep neural network architectures and weights, and training code for the [BASE paper](https://arxiv.org/abs/1812.09584).

If you find this useful, or if you use it in your work, please cite:

    @inproceedings{2019_SqueezeNAS,
        author = {Albert Shaw and Wei Wei and Weiyang Liu and Le Song and Bo Dai},
        title = {Meta Architecture Search},
        booktitle = {NeurIPS},
        year = {2019}
    }

## Requirements

```
Python >= 3.6.0
PyTorch >= 1.0.1
torchvision >= 0.2.2
numpy >= 1.15.4
Pillow
```

## Instructions

1. Install the required packages.
1. Clone this repository.
1. Download and extract the Imagenet dataset to `data/imagenet`.

## Evaluation

Use the ```train.py``` script to evaluate the models. Logs are saved into the ```logs``` folder.

Training the networks on cifar10 requires one 1080 TI and 2 1080 TI to train Imagenet.

To evaluate the trained networks run:  
```python3 train.py --model=get_cifar_tuned_model(True) --gpu 1 --eval 1```  
```python3 train_imagenet.py --model=get_imagenet_tuned_model(True) --gpu 1 --eval 1```

To train the found networks run:  
```python3 train.py --model=get_cifar_tuned_model(False) --gpu 1```  
```python3 train_imagenet.py --model=get_imagenet_tuned_model(False) --gpu 1```