###################################
#   Imports                       #
###################################
import torch
import torch.nn as nn
import torchvision
import pandas as pd
import matplotlib.pyplot as plt
import PIL 
import os
import shutil
import time
import numpy as np
import timm
import copy
import glob
import torch.nn.functional as F
from ultralytics import YOLO
from torchvision.models import (
    resnet18, resnet50, 
    vit_b_32, 
    mobilenet_v3_small, 
    vgg11, 
    googlenet, 
    convnext_tiny,convnext_small, convnext_base,
    efficientnet_v2_s, efficientnet_v2_m,
    inception_v3,
    densenet121,
    ResNet18_Weights, ResNet50_Weights, 
    ViT_B_32_Weights, 
    MobileNet_V3_Small_Weights, 
    VGG11_Weights, 
    GoogLeNet_Weights,
    ConvNeXt_Tiny_Weights, ConvNeXt_Small_Weights, ConvNeXt_Base_Weights,
    EfficientNet_V2_S_Weights, EfficientNet_V2_M_Weights,
    Inception_V3_Weights,
    DenseNet121_Weights,
    )
from torchvision import datasets, transforms
from torchvision.transforms import v2
from transformers import AutoModel
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score ,confusion_matrix, ConfusionMatrixDisplay
from PIL import Image, ExifTags
from collections import Counter
from ultralytics import YOLO
from multiprocessing import freeze_support
from adan_pytorch import Adan




###################################
#   Variaveis                     #
###################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 4

SEED = [42, 43, 44, 45]

###################################
#   Definições                    #
###################################
transforms_0 = v2.Compose([
    v2.Resize(size=(224, 224)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model_names = ["resnet18", "resnet50", "vit", "mobilenet_v3_small", "googlenet", "dino_vit", "vgg11", "convnext_tiny", "convnext_small", "convnext_base", "efficientnet_v2_s", "efficientnet_v2_m", "densenet121", "se_resnext50","nasnet_large", "YOLOm", "YOLOn"]
model_names = ["YOLOm", "YOLOn"]
#"resnet18", "googlenet","vgg11","dino_vit",,"vit",
#

BASE_CONFIG = {
    "learning_rate": 1e-3,
    "optimizer": "AdamW",
    "loss_type": "crossEntropy",
    "scheduler": None,
    "train_transform" : transforms_0,
    "test_transform" : transforms_0,
}

modelos_config = {
    name: BASE_CONFIG.copy()
    for name in model_names
}


###################################
#   Métodos de data augmentation  #
###################################

training_transforms_1 = v2.Compose([
    v2.Resize(size=(224, 224)),
    v2.RandomRotation(10),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

training_transforms_2 = v2.Compose([
    v2.Resize(size=(224, 224)),
    v2.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2,hue=0.05),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

training_transforms_3 = v2.Compose([
    v2.RandomResizedCrop(224, scale=(0.8, 1.0)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

training_transforms_4 = v2.Compose([
    v2.Resize(size=(224, 224)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

training_transforms_5 = v2.Compose([
    v2.Resize(size=(224, 224)),
    v2.RandomAffine(degrees=10,translate=(0.1, 0.1),scale=(0.9, 1.1),shear=5),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

training_transforms_6 = v2.Compose([
    v2.Resize(size=(224, 224)),
    v2.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


##############

training_transforms_final1 = v2.Compose([
    v2.RandomResizedCrop(224, scale=(0.8,1.0)),
    v2.RandomRotation(10),
    v2.RandomAffine(degrees=10,translate=(0.1, 0.1),scale=(0.9, 1.1),shear=5),
    v2.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485,0.456,0.406],
                 std=[0.229,0.224,0.225])
])

training_transforms_final2 = v2.Compose([
    v2.RandomResizedCrop(224, scale=(0.8,1.0)),
    v2.RandomRotation(10),
    v2.RandomAffine(degrees=10,translate=(0.1, 0.1),scale=(0.9, 1.1),shear=5),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485,0.456,0.406],
                 std=[0.229,0.224,0.225])
])

training_transforms_final3 = v2.Compose([
    v2.RandomResizedCrop(224, scale=(0.9,1.0)),
    v2.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

training_transforms_final4 = v2.Compose([
    v2.RandomResizedCrop(224, scale=(0.9,1.0)),
    v2.RandomAffine(degrees=10,translate=(0.1, 0.1),scale=(0.9, 1.1),shear=5),

    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485,0.456,0.406],
                 std=[0.229,0.224,0.225])
])
#################

TRANSFORMS = {
    "controlo": transforms_0,
    "augmentation_1": training_transforms_1,
    "augmentation_2": training_transforms_2,
    "augmentation_3": training_transforms_3,
    "augmentation_4": training_transforms_4,
    "augmentation_5": training_transforms_5,
    "augmentation_6": training_transforms_6,
    "augmentation_final1": training_transforms_final1,
    "augmentation_final2": training_transforms_final2,
    "augmentation_final3": training_transforms_final3,
    "augmentation_final4": training_transforms_final4,
}
