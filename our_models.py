from pathlib import Path
from typing import NamedTuple, List, Tuple

import torch

from model import NetworkCIFAR, NetworkImageNet, CrossEntropyLabelSmooth

CIFAR_CLASSES = 10
IMAGENET_CLASSES = 1000

class Genotype(NamedTuple):
    normal: List[Tuple[str, int]]
    reduce: List[Tuple[str, int]]


CIFAR_TUNED = Genotype(normal=[('skip_connect', 0), ('dil_conv_3x3', 1),
                               ('max_pool_3x3', 0), ('dil_conv_5x5', 0),
                               ('max_pool_3x3', 0), ('sep_conv_5x5', 0),
                               ('max_pool_3x3', 1), ('dil_conv_5x5', 1)],
                       reduce=[('sep_conv_3x3', 0), ('sep_conv_5x5', 0),
                               ('max_pool_3x3', 0), ('sep_conv_3x3', 0),
                               ('skip_connect', 0), ('avg_pool_3x3', 3),
                               ('max_pool_3x3', 1), ('skip_connect', 2)])

IMAGENET_TUNED = Genotype(normal=[('skip_connect', 0), ('max_pool_3x3', 1),
                                  ('avg_pool_3x3', 0), ('sep_conv_5x5', 1),
                                  ('sep_conv_5x5', 0), ('dil_conv_5x5', 3),
                                  ('sep_conv_5x5', 0), ('sep_conv_5x5', 2)],
                          reduce=[('max_pool_3x3', 0), ('sep_conv_5x5', 0),
                                  ('max_pool_3x3', 0), ('avg_pool_3x3', 0),
                                  ('sep_conv_5x5', 2), ('dil_conv_5x5', 2),
                                  ('avg_pool_3x3', 0), ('sep_conv_5x5', 3)])


def get_cifar_tuned_model(load_weights=True):
    network = NetworkCIFAR(40, CIFAR_CLASSES, 20, True, 0.4, CIFAR_TUNED)
    if load_weights:
        device = torch.device('cpu')
        state_dict = torch.load('weights/cifar_tuned.pt', map_location=device)
        network.load_state_dict(state_dict)
    return network

def get_imagenet_tuned_model(load_weights=True):
    network = NetworkImageNet(48, IMAGENET_CLASSES, 14, True, 0.4, IMAGENET_TUNED, CrossEntropyLabelSmooth(IMAGENET_CLASSES, 0.1))
    if load_weights:
        device = torch.device('cpu')
        state_dict = torch.load('weights/imagenet_tuned.pt', map_location=device)
        # state_dict = {k:v for k,v in state_dict.items() if not 'total_ops' in k and not 'total_params' in k}
        network.load_state_dict(state_dict)
    return network