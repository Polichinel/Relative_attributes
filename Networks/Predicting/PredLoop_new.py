# use torch_2022

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


# models - you do need the weights for the transformations.
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision.models import regnet_x_8gf, RegNet_X_8GF_Weights
from torchvision.models import swin_t, Swin_T_Weights
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights

#import wandb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
print(sys.version)
print(torch.__version__)
print(torchvision.__version__)


def get_img_id(img_dir):

    """Creates a list of all image ids/names."""

    img_name_list = []
    
    for root, dirs, files in os.walk(img_dir):
        for img_name in files:
            if img_name.split('.')[1] == 'jpg':
                img_name_list.append(img_name)

    return(img_name_list)


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):

        self.img_id = get_img_id(img_dir)
        self.img_dir = img_dir
        self.transform = transform

        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_id)

    def __getitem__(self, idx):
        img_id = self.img_id[idx]
        img_path = os.path.join(self.img_dir, img_id)
        image = read_image(img_path)
        
        #Better than the try
        if image.shape[0] == 1:
            image = image.expand(3, -1, -1)


        if self.transform:
            image = self.transform(image)

        return image, img_id


def change_head(model_name, model, num_classes):

    if model_name == 'convnext_tiny':
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes, bias=False).to(device)
        #print(f'new head: {model.classifier[2]}')

    elif model_name == 'efficientnet_v2_s':
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes, bias=False).to(device)
        #print(f'new head: {model.classifier[1]}')

    elif model_name == 'regnet_x_8gf':
        model.fc = nn.Linear(model.fc.in_features, num_classes, bias=False).to(device)
        #print(f'new head: {model.fc}')

    elif model_name == 'swin_t':
        model.head = nn.Linear(model.head.in_features, num_classes, bias=False).to(device)
        #print(f'new head: {model.head}')

    elif model_name == 'wide_resnet50_2':
        model.fc = nn.Linear(model.fc.in_features, num_classes, bias=False).to(device)
        #print(f'new head: {model.fc}')

    elif model_name == 'squeezenet1_1' :
        model.classifier[1] = nn.Conv2d(model.classifier[1].in_channels, num_classes, kernel_size=(1, 1), stride=(1, 1), bias=False).to(device)
        #print(f'new head: {model.classifier[1]}')

    elif model_name == 'shufflenet_v2_x0_5' :
        model.fc = nn.Linear(model.fc.in_features, num_classes, bias=False).to(device)
        #print(f'new head: {model.fc}')

    elif model_name == 'mnasnet0_5' :
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes, bias=False).to(device)
        #print(f'new head: {model.classifier[1]}')    

    elif model_name == 'mobilenet_v3_small' :
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes, bias=False).to(device)
        #print(f'new head: {model.classifier[3]}')

    else:
        print('Unddefined model name...')


# JUST LOAD PRETRAINED MODEL

def get_model(hyperparameters):

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

    # model_name = hyperparameters['model_name']

    model_name = hyperparameters['model_name']
    attribute = hyperparameters['attribute']

    #pt_model_dir = "/home/simon/Documents/computerome/done_RA_models/" #!!!!!
    pt_model_dir = "/home/projects/ku_00017/people/simpol/scripts/bodies/Relative_attributes/Networks/Done_models/"
    pt_model_name = f'{model_name}_{attribute}'

    model = model_dict[model_name]
    change_head(model_name, model, 1)
    
    model.load_state_dict(torch.load(f'{pt_model_dir}{pt_model_name}_SD.pth')) #, map_location=torch.device('cpu'))) #!!!! Remove or change map_location
    model.eval()

    return(model, weight_dict)


    # data loader
def make_loader(batch_size, weights ): #, attribute_dict):

    data_transform = transforms.Compose([weights.transforms()])

    # img_dir = '/media/simon/Seagate Expansion Drive/images_spanner' #local
    img_dir = '/home/projects/ku_00017/data/raw/bodies/images_spanner' # computerome
    #img_dir = '/home/projects/ku_00017/data/raw/bodies/images_spanner' # computerome


    image_dataset = CustomImageDataset(img_dir, transform=data_transform)
    dataloader = DataLoader(image_dataset, batch_size=batch_size, shuffle=False)

    dataset_size = len(image_dataset)

    return dataloader, dataset_size


def make(hyperparameters):

    # Make the model
    model, weight_dict = get_model(hyperparameters)

    model_name = hyperparameters['model_name']

    #Choose model and wieghts
    weights = weight_dict[model_name] # you need these later right as they hold the appropiate data transforamtions
    model = model.to(device)
    

    # Make the data
    dataloader, dataset_size = make_loader(batch_size=hyperparameters['batch_size'],  weights = weights) #, attribute_dict = attribute_dict)

    return model, dataloader, dataset_size


def predict(model, dataloader):
    model.eval() # one more time why not.

    image_list = []
    score_list = []

    # Run the model on some test examples
    with torch.no_grad():
        count = 1
        
        for images, img_id in dataloader:

            print(f'{img_id}, {count}/{dataloader.__len__()}.............', end='\r')
                
            images = images.to(device)
            
            outputs = model(images)

            image_list.append(img_id)
            score_list.append(outputs.detach().cpu().numpy())

            count +=1
            
    return(image_list, score_list)



def the_loop(model_name):

    attributes = ['all_negative_emotions_t1', 'all_mass_protest', 'all_militarized',
                'all_urban', 'all_negative_emotions_t2', 'all_privat', 'all_public', 
                'all_rural', 'all_formal', 'all_damaged_property']

    data_dir = '/home/projects/ku_00017/data/generated/bodies/ra_outputs/'

    print(f'{model_name} running...')
    
    score_dict = {}
    
    for i, attribute in enumerate(attributes):
        print(f'predicting {attribute}, {i}/{len(attributes)}')

        hyperparameters = {"model_name" : model_name, "attribute" : attribute, "batch_size": 1} # Just do one at a time...
        model, dataloader, dataset_size = make(hyperparameters)

        print(f' {dataset_size}')

        image_list, score_list = predict(model, dataloader)

        score_dict[f'{attribute}_score'] = score_list,
        score_dict[f'{attribute}_id'] = image_list # not sure you get the rigth order so this is just for debug really.

        # running backup
        attribute_tuple = (image_list, score_list)
        tuple_name = f'{model_name}_{attribute}_tuple.pkl'
        with open(f'{data_dir}{tuple_name}', 'wb') as file:
            pickle.dump(attribute_tuple, file)

        print(f'Backup {tuple_name} pickled')

    dict_name = f'{model_name}_score_dict_full.pkl'

    with open(f'{data_dir}{dict_name}', 'wb') as file:
        pickle.dump(score_dict, file)

    print('Pickled and done!')



if __name__ == "__main__":

    model_letter = input('Choose model: a) convnext, b) efficient or c) swin')
    
    if model_letter == 'a':
        model_name = 'convnext_tiny'

    elif model_letter == 'b':
        model_name = 'efficientnet_v2_s'

    elif model_letter == 'c':
        model_name = 'swin_t'

    else:
        print('wrong input...')
        sys.exit

    the_loop(model_name)
