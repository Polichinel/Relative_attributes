import numpy as np
#import pandas as pd
import pickle
import time
import os
import copy
import sys
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms

# models
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision.models import regnet_x_8gf, RegNet_X_8GF_Weights
from torchvision.models import swin_t, Swin_T_Weights
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights

import wandb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
print(sys.version)
print(torch.__version__)
print(torchvision.__version__)


# Fuctions - could go in own script

#Get the models
def get_models():

    weight_dict = {'convnext_tiny': ConvNeXt_Tiny_Weights.DEFAULT,
                'efficientnet_v2_s' : EfficientNet_V2_S_Weights.DEFAULT,
                'regnet_x_8gf' : RegNet_X_8GF_Weights.DEFAULT,
                'swin_t' : Swin_T_Weights.DEFAULT,
                'wide_resnet50_2' : Wide_ResNet50_2_Weights.DEFAULT}

    model_dict = {'convnext_tiny': convnext_tiny(weights = weight_dict['convnext_tiny']).to(device),
                'efficientnet_v2_s' : efficientnet_v2_s(weights = weight_dict['efficientnet_v2_s']).to(device),
                'regnet_x_8gf' : regnet_x_8gf(weights = weight_dict['regnet_x_8gf']).to(device),
                'swin_t' : swin_t(weights = weight_dict['swin_t']).to(device),
                'wide_resnet50_2' : wide_resnet50_2(weights = weight_dict['wide_resnet50_2']).to(device)}

    return(weight_dict, model_dict)


# change last layer
def change_head(model_name, model, num_classes):

    if model_name == 'convnext_tiny':
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes).to(device)
        print(f'new head: {model.classifier[2]}')

    elif model_name == 'efficientnet_v2_s':
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes).to(device)
        print(f'new head: {model.classifier[1]}')

    elif model_name == 'regnet_x_8gf':
        model.fc = nn.Linear(model.fc.in_features, num_classes).to(device)
        print(f'new head: {model.fc}')

    elif model_name == 'swin_t':
        model.head = nn.Linear(model.head.in_features, num_classes).to(device)
        print(f'new head: {model.head}')

    elif model_name == 'wide_resnet50_2':
        model.fc = nn.Linear(model.fc.in_features, num_classes).to(device)
        print(f'new head: {model.fc}')

    else:
        print('Unddefined model name...')


# data loader
def make_loader(batch_size, weights):

    # this is the thiong that has to change for RA...

    data_transforms = {
    'train': transforms.Compose([weights.transforms(), transforms.RandomHorizontalFlip()]), 
    'val': transforms.Compose([weights.transforms()])
    }

    # Load data - a lot needs to change here since you have a score for each image and not a class (given by dir)
    # data_dir = '/home/simon/Documents/Bodies/data/RA/Tutorial/hymenoptera_data' #local
    data_dir = '/home/projects/ku_00017/data/raw/beesNants/hymenoptera_data' # computerome


    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    return dataloaders, dataset_sizes, class_names

def make(config, model_name):

    # Make the model
    weight_dict, model_dict = get_models()

    #Choose model and wieghts
    weights = weight_dict[model_name] # you need these later right as they hold the appropiate data transforamtions
    model = model_dict[model_name].to(device)
    # wandb.watch(model)

    # new model head for for retraining
    change_head(model_name, model, config['classes'])

    # re-train all parameters
    for param in list(model.parameters()):
        param.requires_grad = True
    
    # Make the data
    dataloaders, dataset_sizes, class_names = make_loader(batch_size=config.batch_size, weights = weights)

    # Make the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay = config.weight_decay)

    return model, criterion, optimizer, dataloaders, dataset_sizes, class_names


def train_log(loss, example_ct, epoch):
    # Where the magic happens
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")


def train_batch(images, labels, model, optimizer, criterion):
    images, labels = images.to(device), labels.to(device)
    
    # Forward pass ➡
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    # Backward pass ⬅
    optimizer.zero_grad()
    loss.backward()

    # Step with optimizer
    optimizer.step()

    return loss


def train(model, loader, criterion, optimizer, config):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)

    # Run training and track with wandb
    total_batches = len(loader) * config.epochs
    example_ct = 0  # number of examples seen
    batch_ct = 0

    for epoch in range(config.epochs):
        for _, (images, labels) in enumerate(loader):

            loss = train_batch(images, labels, model, optimizer, criterion)
            example_ct +=  len(images)
            batch_ct += 1

            running_loss += loss

            # Report metrics every 20th batch - not running average right now
            if ((batch_ct + 1) % 10) == 0:
                #train_log(loss, example_ct, epoch) # this is the wand part
                train_log(running_loss/10, example_ct, epoch)
                running_loss = 0.0 # reset


def test(model, test_loader):
    model.eval()

    # Run the model on some test examples
    with torch.no_grad():
        correct, total = 0, 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Accuracy of the model on the {total} " +
              f"test images: {100 * correct / total}%")
        
        wandb.log({"test_accuracy": correct / total})

    # Save the model in the exchangeable ONNX format
    torch.onnx.export(model, images, "model.onnx")
    wandb.save("model.onnx")


def model_pipeline(hyperparameters):

    # tell wandb to get started
    with wandb.init(project="test_project_0", entity="nornir", config=hyperparameters):
      # access all HPs through wandb.config, so logging matches execution!
      config = wandb.config

      model_name = config['model_name']

      # make the model, data, and optimization problem
      model, criterion, optimizer, dataloaders, dataset_sizes, class_names = make(config, model_name)
      print(model)

      # and use them to train the model
      train(model, dataloaders['train'], criterion, optimizer, config)

      # and test its final performance
      test(model, dataloaders['val'])

    return model



if __name__ == "__main__":

    wandb.login()
    #wandb.init(project="test_project_0", entity="nornir")

    # choose one model
    # model_name = 'convnext_tiny' # last block is called "classifier" 
    model_name = 'efficientnet_v2_s' # last block is called "classifier" 
    # model_name = 'regnet_x_8gf' # last block is called "fc" 
    # model_name = 'swin_t'  # last block is called "head" 
    # model_name = 'wide_resnet50_2'  # last block is called "fc" 

    hyperparameters = {
    "model_name" : model_name,
    "learning_rate": 0.001,
    "weight_decay" : 0.01,
    "classes" : 2,
    "epochs": 32,
    "batch_size": 16
    }

    # Build, train and analyze the model with the pipeline
    model = model_pipeline(hyperparameters)