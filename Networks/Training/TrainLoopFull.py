import numpy as np
#import pandas as pd
import pickle
import time
import os
import copy
import sys
#import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, models, transforms
from torchvision.io import read_image


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

# you custom RA dataset class
class CustomImageDataset(Dataset):
    def __init__(self, attribute_dict, attribute, img_dir, transform=None, target_transform=None):

        self.img_labels = np.array(list(zip(attribute_dict['img'], attribute_dict[f'{attribute}_ens_mean']))) # which is not a label but a score.. # you do not handles nans right now... 
        self.img_std = np.array(list(zip(attribute_dict['img'], attribute_dict[f'{attribute}_ens_std']))) # you do not handles nans right now...
        
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx, 0])
        image = read_image(img_path)
        label = np.float32(self.img_labels[idx, 1]) #it is turned to a str above so we turn it back here
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


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
def make_loader(batch_size, weights, attribute):

    # this is the thiong that has to change for RA...
    # need to get attribute

    #Should be in config
    data_transforms = transforms.Compose([weights.transforms(), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(degrees=(0, 45)), transforms.ColorJitter(brightness=.5, hue=.2)])
    

    # Load data - a lot needs to change here since you have a score for each image and not a class (given by dir)
    # data_dir = '/home/simon/Documents/Bodies/data/RA/Tutorial/hymenoptera_data' #local
    # data_dir = '/home/projects/ku_00017/data/raw/beesNants/hymenoptera_data' # computerome


    # Going into make loader -------------------------------

    #dict_dir = '/home/simon/Documents/Bodies/data/RA/dfs/' #local
    #img_dir = '/media/simon/Seagate Expansion Drive/images_spanner' #local

    dict_dir = '/home/projects/ku_00017/data/raw/bodies/RA_annotations/' # computerome
    img_dir = '/home/projects/ku_00017/data/raw/bodies/images_spanner' # computerome

    with open(f'{dict_dir}ra_ens_annotated_dict.pkl', 'rb') as file:
        attribute_dict = pickle.load(file)

    #dataloaders = {}
    #image_datasets = {}
    
    # CHANGE!!------------------------
    image_dataset = CustomImageDataset(attribute_dict, attribute, img_dir, transform=data_transforms)
    dataloader = DataLoader(image_dataset, batch_size=batch_size, shuffle=True)

    dataset_size = len(image_dataset)
    #class_names = image_datasets['train'].classes

    return dataloader, dataset_size #class_names # what did you use class names for?

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
    #dataloaders, dataset_sizes, class_names = make_loader(batch_size=config.batch_size, weights = weights)
    dataloaders, dataset_size = make_loader(batch_size=config.batch_size,  weights = weights, attribute = config.attribute)
    
    # Make the loss and optimizer
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay = config.weight_decay, betas = config.betas)

    return model, criterion, optimizer, dataloaders, dataset_size #, class_names

def train_log(loss, example_ct, epoch):
    # Where the magic happens
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")


def train_batch(images, labels, model, optimizer, criterion):
    images, labels = images.to(device), labels.to(device)
    
    # Forward pass ➡
    outputs = model(images)
    loss = criterion(outputs.squeeze(), labels)
    
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
    running_loss = 0.0

    for epoch in range(config.epochs):
        for _, (images, labels) in enumerate(loader):

            loss = train_batch(images, labels, model, optimizer, criterion)
            example_ct +=  len(images)
            batch_ct += 1

            running_loss += loss

            # Report metrics every 20th batch - not running average right now
            if ((batch_ct + 1) % 100) == 0:
                #train_log(loss, example_ct, epoch) # this is the wand part
                train_log(running_loss/100, example_ct, epoch)
                running_loss = 0.0 # reset



# CHANGE!!------------------------ this will just be full insample
def test(model, test_loader):
    model.eval()
    test_criterion = nn.MSELoss()

    # Run the model on some test examples
    with torch.no_grad():
        #correct, total = 0, 0
        total = 0
        RMSE_loss = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            #_, predicted = torch.max(outputs.data, 1)
            RMSE_loss += torch.sqrt(test_criterion(outputs.squeeze().cpu(), labels.cpu()))

            total += labels.size(0)
            #correct += (predicted == labels).sum().item()

        # print(f"Accuracy of the model on the {total} " +
        #       f"test images: {100 * correct / total}%")
        
        print(f"Average RMSE of the model on the {total} " +
              f"test images: {RMSE_loss / total}%")


        wandb.log({"final_rmse": RMSE_loss / total})

    # Save the model in the exchangeable ONNX format
    torch.onnx.export(model, images, "model.onnx")
    wandb.save("model.onnx")


def model_pipeline(hyperparameters):

    # tell wandb to get started
    with wandb.init(project="RA_full", entity="nornir", config=hyperparameters): #new projrct name!!!
      # access all HPs through wandb.config, so logging matches execution!
      config = wandb.config

      model_name = config['model_name']

      # make the model, data, and optimization problem
      model, criterion, optimizer, dataloader, dataset_size = make(config, model_name)
      print(model)

      # CHANGE!!------------------------
      # and use them to train the model
      train(model, dataloader, criterion, optimizer, config)

      # CHANGE!!------------------------
      # and test its final performance
      test(model, dataloader)

    return model



if __name__ == "__main__":

    wandb.login()

    input_dict1 = {'a': 'convnext_tiny',
                  'b': 'efficientnet_v2_s',
                  'c': 'regnet_x_8gf',
                  'd' : 'swin_t',
                  'e' : 'wide_resnet50_2'}

    model_string = f"Choose model:\n "
    for k in input_dict1.keys():
        model_string += f"{k}) {input_dict1[k]}\n "

    print(model_string)

    input_string1 = input()
    if input_string1 in ['a', 'b', 'c', 'd', 'e']:
        model_name = input_dict1[input_string1]
        print(f'You choose {input_string1} : {model_name}')

    else:
        print('Wrong input')
        exit()

    input_dict2 = {'a' : 'all_negative_emotions_t1',
                   'b' : 'all_mass_protest_ens',
                   'c' : 'all_militarized',
                   'd' : 'all_urban',
                   'e' : 'all_negative_emotions_t2',
                   'f' : 'all_privat',
                   'g' : 'all_public',
                   'h' : 'all_rural',
                   'i' : 'all_formal',
                   'j' : 'all_damaged_property'}


    att_string = f"Choose attribute:\n "
    for k in input_dict2.keys():
        att_string += f"{k}) {input_dict2[k]}\n "

    print(att_string)

    input_string2 = input()
    if input_string2 in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']:
        attribute = input_dict2[input_string2]
        print(f'You choose {input_string2} : {attribute}')

    else:
        print('Wrong input')
        exit()
    # ---------------------------------------

    hyperparameters = {
    "model_name" : model_name,
    "attribute" : attribute,
    "learning_rate": 0.0001,
    "weight_decay" : 0.05,
    'betas' : (0.9, 0.999),
    "classes" : 1,
    "epochs": 32,
    "batch_size": 64
    }

    # Build, train and analyze the model with the pipeline
    model = model_pipeline(hyperparameters)

    # save model and weights - computerome path.
    PATH = f"/home/projects/ku_00017/people/simpol/scripts/bodies/Relative_attributes/Networks/Done_models/{model_name}_{attribute}.pth"
    torch.save(model.state_dict(), PATH)