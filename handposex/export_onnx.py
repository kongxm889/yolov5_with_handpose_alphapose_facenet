import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import sys

from utils.model_utils import *
from utils.common_utils import *
from hand_data_iter.datasets import *

from models.resnet import resnet50, resnet101
from models.squeezenet import squeezenet1_1, squeezenet1_0
from torchvision.models.shufflenetv2 import shufflenet_v2_x1_0
from torchvision.models.shufflenetv2 import shufflenet_v2_x1_5
from torchvision.models.shufflenetv2 import shufflenet_v2_x2_0

from loss.loss import *
import cv2
import time
import json
from datetime import datetime


def export():
    if ops.model == 'resnet_50':
        model_ = resnet50(pretrained=True, num_classes=ops.num_classes,
                            img_size=ops.img_size[0], dropout_factor=ops.dropout)
    elif ops.model == 'resnet_34':
        model_ = resnet34(pretrained=True, num_classes=ops.num_classes,
                            img_size=ops.img_size[0], dropout_factor=ops.dropout)
    elif ops.model == 'resnet_101':
        model_ = resnet101(pretrained=True, num_classes=ops.num_classes,
                            img_size=ops.img_size[0], dropout_factor=ops.dropout)
    elif ops.model == "squeezenet1_0":
        model_ = squeezenet1_0(
            pretrained=True, num_classes=ops.num_classes, dropout_factor=ops.dropout)
    elif ops.model == "squeezenet1_1":
        model_ = squeezenet1_1(
            pretrained=True, num_classes=ops.num_classes, dropout_factor=ops.dropout)
    elif ops.model == 'shufflenetv2_x1_0':
        model_ = shufflenet_v2_x1_0(
            pretrained=False, num_classes=ops.num_classes)
    elif ops.model == 'shufflenetv2_x1_5':
        model_ = shufflenet_v2_x1_5(
            pretrained=False, num_classes=ops.num_classes)
    else:
        print(" no support the model")

    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda:0" if use_cuda else "cpu")
    model_ = model_.to(device)


if __name__ == '__main__':
    export()