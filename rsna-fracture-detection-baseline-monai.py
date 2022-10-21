#!/usr/bin/env python
# coding: utf-8

## Load Packages

!pip install warmup_scheduler
!pip install -U pydicom
!pip install albumentations
!pip install monai


import os
import sys
import re
import gc
from pathlib import Path
import random
import math
import shutil
from glob import glob
from tqdm import tqdm
from pprint import pprint
from time import time
import warnings
import itertools
import pandas as pd
import numpy as np
import multiprocessing as mp
import cv2
import PIL
from PIL import Image

# .dcm handling
import pydicom

# PyTorch
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
import torchvision 
import torchvision.transforms as transforms
from warmup_scheduler import GradualWarmupScheduler
import albumentations

# MONAI 3D
from monai.transforms import Randomizable, apply_transform
from monai.transforms import Compose, Resize, ScaleIntensity, ToTensor, RandAffine
from monai.networks.nets import densenet

from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

# Environment check
warnings.filterwarnings("ignore")

### Helper Function
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

# # some patients have reverse order for the CT scan, so have a function to check
# def check_reverse_required(path):
#     paths = list(path.glob('*'))
#     paths.sort(key=lambda x:int(x.stem))
#     z_first = pydicom.dcmread(paths[0]).get("ImagePositionPatient")[-1]
#     z_last = pydicom.dcmread(paths[-1]).get("ImagePositionPatient")[-1]
#     if z_last < z_first:
#         return False
#     return True

# paths = {
#     'train': Path('../input/rsna-2022-cervical-spine-fracture-detection/train.csv'),
#     'train_bbox': Path('../input/rsna-2022-cervical-spine-fracture-detection/train_bounding_boxes.csv'),
#     'train_images': Path('../input/rsna-2022-cervical-spine-fracture-detection/train_images'),
#     'train_nifti_segments': Path('../input/rsna-2022-cervical-spine-fracture-detection/segmentations'),
#     'test_df': Path('../input/rsna-2022-cervical-spine-fracture-detection/test.csv'),
#     'test_images': Path('../input/rsna-2022-cervical-spine-fracture-detection/test_images')
# }



# custom weighted loss function
# From: https://www.kaggle.com/code/andradaolteanu/rsna-fracture-detect-pytorch-densenet-train#2.-Data-Split
def get_custom_loss(logits, targets):
    
    # Compute the weights
    weights = targets * competition_weights['+'] + (1 - targets) * competition_weights['-']
    
    # Losses on label and exam level
    L = torch.zeros(targets.shape, device=DEVICE)

    w = weights
    y = targets
    p = logits
    eps=1e-8

    for i in range(L.shape[0]):
        for j in range(L.shape[1]):
            L[i, j] = -w[i, j] * (
                y[i, j] * math.log(p[i, j] + eps) +
                (1 - y[i, j]) * math.log(1 - p[i, j] + eps))
            
    # Average Loss on Exam (or patient)
    Exams_Loss = torch.div(torch.sum(L, dim=1), torch.sum(w, dim=1))
    
    return Exams_Loss

class RSNADataset(Dataset, Randomizable):
    
    def __init__(self, csv, mode, transform=None):
        self.csv = csv
        self.mode = mode
        self.transform = transform
        
    def __len__(self):
        return self.csv.shape[0]
    
    def randomize(self) -> None:
        '''-> None is a type annotation for the function that states 
        that this function returns None.'''
        
        MAX_SEED = np.iinfo(np.uint32).max + 1
        self.seed = self.R.randint(MAX_SEED, dtype="uint32")
        
    def __getitem__(self, index):
        # Set Random Seed
        self.randomize()
        
        dt = self.csv.iloc[index, :]
        study_paths = glob(f"{DATA_PATH}/train_images/{dt.StudyInstanceUID}/*")
        study_paths.sort(key=natural_keys)
        
        # Load images
        study_images = [cv2.imread(path)[:,:,::-1] for path in study_paths]
        
        # Stack all scans into 1
        stacked_image = np.stack([img.astype(np.float32) for img in study_images], 
                                 axis=2).transpose(3,0,1,2)
        
        if self.transform:
            if isinstance(self.transform, Randomizable):
                self.transform.set_random_state(seed=self.seed)
                
            stacked_image = apply_transform(self.transform, stacked_image)
        
        if self.mode=="test":
            return {"image": stacked_image}
        else:
            targets = torch.tensor(dt[target_cols]).float()
            return {"image": stacked_image,
                    "targets": targets}

# send the data to GPU
def data_to_device(data):
    image, targets = data.values()
    return image.to(DEVICE), targets.to(DEVICE)

## Loss & Gradual warmup

# Reference link: [HERE](https://stackoverflow.com/questions/42479902/what-does-view-do-in-pytorch)
# 
# torch.view(-1):
# * view() reshapes the tensor without copying memory, similar to numpy's reshape().
# * -1 flatten the tensor

def get_criterion(logits, target): 
    loss = CRITERION(logits.view(-1), target.view(-1))
    return loss


class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    '''
    src: https://www.kaggle.com/code/boliu0/monai-3d-cnn-training/notebook
    '''
    
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(optimizer, multiplier, 
                                                       total_epoch, after_scheduler)
    
    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier 
                                                     for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) 
                    for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) 
                    for base_lr in self.base_lrs]

## Log the info
def add_in_file(text, f):
    with open(f'log_{KERNEL_TYPE}.txt', 'a+') as f:
        print(text, file=f)

## Train Epoch
def train_epoch(model, dataloader, optimizer, epoch, f):
    
    # Add info to file
    print("Training...")
    add_in_file('Training...', f)
    
    # Track training time for 1 epoch
    start_time = time()
    
    # === TRAIN ===
    model.train()
    train_losses, train_comp_losses = [], []
    
    # Loop through the data
    bar = tqdm(dataloader)
    for data in bar:
        image, targets = data_to_device(data)
        
        # Train & Optimize
        optimizer.zero_grad()
        logits = model(image)
        loss = get_criterion(logits, targets)
        loss.sum().backward()
        optimizer.step()
        
        # === COMP LOSS ===
        comp_loss = get_custom_loss(logits, targets)

        # Save losses
        train_losses.append(loss.detach().cpu().numpy())
        train_comp_losses.append(comp_loss.detach().cpu().numpy().mean())
        
        gc.collect()

    # Compute Overall Loss
    mean_train_loss = np.mean(train_losses)
    mean_comp_loss = np.mean(train_comp_losses)
    
    # Save info
    total_time = round((time() - start_time)/60, 3)
    add_in_file('Train Mean Loss: {}'.format(mean_train_loss), f)
    add_in_file('Train Mean Comp Loss: {}'.format(mean_comp_loss), f)
    add_in_file('~~~ Train Time: {} mins ~~~'.format(total_time), f)
                
    # Print info
    print("Train Mean Loss:", mean_train_loss)
    print("Train Mean Comp Loss:", mean_comp_loss)
    print(f"~~~ Train Time: {total_time} mins ~~~")
    
    return mean_train_loss

## Validation Epoch
def valid_epoch(model, dataloader, epoch, f):
    
    # Add info to file
    print("Validation...")
    add_in_file('Validation...', f)
    
    # Track validation time for 1 epoch
    start_time = time()
    
    # === EVAL ===
    model.eval()
    valid_preds, valid_targets, valid_comp_loss = [], [], []
    
    with torch.no_grad():
        for data in dataloader:
            
            image, targets = data_to_device(data)
            logits = model(image)
            
            # === COMP LOSS ===
            comp_loss = get_custom_loss(logits, targets)
            # Save actuals, preds and losses
            valid_targets.append(targets.detach().cpu())
            valid_preds.append(logits.detach().cpu())
            valid_comp_loss.append(comp_loss.detach().cpu().numpy().mean())
            
            gc.collect()
            
    # Overall Valid Loss
    valid_losses = get_criterion(torch.cat(valid_preds), torch.cat(valid_targets)).numpy()
    mean_valid_loss = np.mean(valid_losses)
    
    # Overall Competition Loss
    mean_comp_valid_loss = np.mean(valid_comp_loss)
    
    # Compute Area Under Curve
    PREDS = np.concatenate(torch.cat(valid_preds).numpy())
    TARGETS = np.concatenate(torch.cat(valid_targets).numpy())
    auc = roc_auc_score(TARGETS, PREDS)
    
    # Save info
    total_time = round((time() - start_time)/60, 3)
    add_in_file('Valid Mean Loss: {}'.format(mean_valid_loss), f)
    add_in_file('Valid Mean Comp Loss: {}'.format(mean_comp_valid_loss), f)
    add_in_file('Valid AUC: {}'.format(auc), f)
    add_in_file('~~~ Valid Time: {} mins ~~~'.format(total_time), f)
        
    # Print info
    print("Valid Mean Loss:", mean_valid_loss)
    print("Valid Mean Comp Loss:", mean_comp_valid_loss)
    print("Valid AUC:", auc)
    print(f"~~~ Validation Time: {total_time} mins ~~~")
    
    return mean_valid_loss

def run_train(fold):
    
    # Get the train and valid data
    train = df[df["fold"] != fold].reset_index(drop=True)
    valid = df[df["fold"] == fold].reset_index(drop=True)
    
    # Create the Dataset & Dataloader
    train_dataset = RSNADataset(csv=train, mode="train", 
                                transform=train_transforms)
    valid_dataset = RSNADataset(csv=valid, mode="train", 
                                transform=valid_transforms)
    trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                             sampler=RandomSampler(train_dataset), num_workers=NUM_WORKERS)
    validloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    
    # Model
    model = densenet.densenet121(spatial_dims=3, in_channels=3,
                                 out_channels=OUT_DIM)
    model.class_layers.out = nn.Sequential(nn.Linear(in_features=1024, out_features=OUT_DIM), 
                                           nn.Softmax(dim=1))
    model.to(DEVICE)
    
    # Optimizer & Scheduler
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler_cosine = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 2)
    scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, 
                                                total_epoch=1, 
                                                after_scheduler=scheduler_cosine)
    
    # Initiate initial loss
    valid_loss_BEST = 1000
    # Create model name
    model_file = f'{KERNEL_TYPE}_best_fold{fold}.pth'
    # Create file to save outputs
    f = open(f'log_{KERNEL_TYPE}.txt', 'a')
    
    for epoch in range(EPOCHS):
        
        add_in_file('======== Epoch: {}/{} ========'.format(epoch+1, EPOCHS), f)
        print("="*8, f"Epoch {epoch}", "="*8)
        
        scheduler_warmup.step(epoch-1)
        
        # Train & Validate
        mean_train_loss = train_epoch(model, trainloader, optimizer, epoch, f)
        mean_valid_loss = valid_epoch(model, validloader, epoch, f)
        
        # Save model
        if mean_valid_loss < valid_loss_BEST:
            print('Saving model ...')
            add_in_file('Saving model => {}'.format(model_file), f)
            torch.save(model.state_dict(), model_file)
            valid_loss_BEST = mean_valid_loss
            
    torch.cuda.empty_cache()
    gc.collect()



## Reference

# * [RSNA Fracture Detect: PyTorch DenseNet train](https://www.kaggle.com/code/andradaolteanu/rsna-fracture-detect-pytorch-densenet-train)

if __name__ == "__main__":
    
    # set seed
    set_seed(0)
    
    # set GPU
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", DEVICE)

    ## Set up parameters
    DF_SIZE = .20
    N_SPLITS = 5
    KERNEL_TYPE = 'densenet121_baseline'
    IMG_RESIZE = 150
    STACK_RESIZE = 50
    use_amp = False
    NUM_WORKERS = 0 #mp.cpu_count()
    BATCH_SIZE = 8 #16
    LR = 0.0005
    OUT_DIM = 8
    EPOCHS = 5
    DATA_PATH = "/root/input/rsna-2022-cervical-spine-fracture-detection"

    target_cols = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7','patient_overall']

    competition_weights = {
        '-' : torch.tensor([1, 1, 1, 1, 1, 1, 1, 7], dtype=torch.float, device=DEVICE),
        '+' : torch.tensor([2, 2, 2, 2, 2, 2, 2, 14], dtype=torch.float, device=DEVICE),
    }

    # Load Data
    df = pd.read_csv(f"{DATA_PATH}/train.csv")

    # Sample down df
    instances = df.StudyInstanceUID.unique().tolist()
    instances = random.sample(instances, k=int(len(instances)*DF_SIZE))
    df = df[df["StudyInstanceUID"].isin(instances)].reset_index(drop=True)
    print("Dataframe size:", df.shape)

    # Create folds
    kfold = GroupKFold(n_splits=N_SPLITS)
    df['fold'] = -1

    # Append fold
    for k, (_, valid_i) in enumerate(kfold.split(df,
                                                 groups=df.StudyInstanceUID)):
        df.loc[valid_i, 'fold'] = k
    
    CRITERION = nn.BCEWithLogitsLoss(reduction='none')
    
    # transform
    train_transforms = Compose([ScaleIntensity(), 
                                Resize((IMG_RESIZE, IMG_RESIZE, STACK_RESIZE)), 
                                ToTensor()])
    valid_transforms = Compose([ScaleIntensity(), 
                              Resize((IMG_RESIZE, IMG_RESIZE, STACK_RESIZE)), 
                              ToTensor()])
    
    
    # Train
    run_train(fold=0)