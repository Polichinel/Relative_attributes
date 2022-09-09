import numpy as np
import pickle
import time
import os
import copy
import sys

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

# new models - low parameter models..
from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights
from torchvision.models import shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights
from torchvision.models import mnasnet0_5, MNASNet0_5_Weights
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

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

        self.img_labels = np.array(list(zip(attribute_dict['img'], attribute_dict[f'{attribute}_ens_mean']))) 
        self.img_std = np.array(list(zip(attribute_dict['img'], attribute_dict[f'{attribute}_ens_std']))) 

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
                'wide_resnet50_2' : Wide_ResNet50_2_Weights.DEFAULT,
                'squeezenet1_1' : SqueezeNet1_1_Weights.DEFAULT,
                'shufflenet_v2_x0_5' : ShuffleNet_V2_X0_5_Weights.DEFAULT,
                'mnasnet0_5' : MNASNet0_5_Weights.DEFAULT,
                'mobilenet_v3_small' : MobileNet_V3_Small_Weights.DEFAULT}

    model_dict = {'convnext_tiny': convnext_tiny(weights = weight_dict['convnext_tiny']).to(device),
                'efficientnet_v2_s' : efficientnet_v2_s(weights = weight_dict['efficientnet_v2_s']).to(device),
                'regnet_x_8gf' : regnet_x_8gf(weights = weight_dict['regnet_x_8gf']).to(device),
                'swin_t' : swin_t(weights = weight_dict['swin_t']).to(device),
                'wide_resnet50_2' : wide_resnet50_2(weights = weight_dict['wide_resnet50_2']).to(device),
                'squeezenet1_1' : squeezenet1_1(weights = weight_dict['squeezenet1_1']).to(device),
                'shufflenet_v2_x0_5' : shufflenet_v2_x0_5(weights = weight_dict['shufflenet_v2_x0_5']).to(device),
                'mnasnet0_5' : mnasnet0_5(weights = weight_dict['mnasnet0_5']).to(device),
                'mobilenet_v3_small' : mobilenet_v3_small(weights = weight_dict['mobilenet_v3_small']).to(device),}

    return(weight_dict, model_dict)


# change last layer
def change_head(model_name, model, num_classes):

    if model_name == 'convnext_tiny':
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes, bias=False).to(device)
        print(f'new head: {model.classifier[2]}')

    elif model_name == 'efficientnet_v2_s':
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes, bias=False).to(device)
        print(f'new head: {model.classifier[1]}')

    elif model_name == 'regnet_x_8gf':
        model.fc = nn.Linear(model.fc.in_features, num_classes, bias=False).to(device)
        print(f'new head: {model.fc}')

    elif model_name == 'swin_t':
        model.head = nn.Linear(model.head.in_features, num_classes, bias=False).to(device)
        print(f'new head: {model.head}')

    elif model_name == 'wide_resnet50_2':
        model.fc = nn.Linear(model.fc.in_features, num_classes, bias=False).to(device)
        print(f'new head: {model.fc}')

    elif model_name == 'squeezenet1_1' :
        model.classifier[1] = nn.Conv2d(model.classifier[1].in_channels, num_classes, kernel_size=(1, 1), stride=(1, 1), bias=False).to(device)
        print(f'new head: {model.classifier[1]}')

    elif model_name == 'shufflenet_v2_x0_5' :
        model.fc = nn.Linear(model.fc.in_features, num_classes, bias=False).to(device)
        print(f'new head: {model.fc}')

    elif model_name == 'mnasnet0_5' :
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes, bias=False).to(device)
        print(f'new head: {model.classifier[1]}')    

    elif model_name == 'mobilenet_v3_small' :
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes, bias=False).to(device)
        print(f'new head: {model.classifier[3]}')

    else:
        print('Unddefined model name...')


# data loader
def make_loader(batch_size, weights, attribute):

    #Should be in config
    data_transforms = {
    'train': transforms.Compose([weights.transforms(), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(degrees=(0, 45)), transforms.ColorJitter(brightness=.25, hue=.1)]), 
    'val': transforms.Compose([weights.transforms()])
    }

    # Going into make loader -------------------------------

    #dict_dir = '/home/simon/Documents/Bodies/data/RA/dfs/' #local
    #img_dir = '/media/simon/Seagate Expansion Drive/images_spanner' #local

    dict_dir = '/home/projects/ku_00017/data/raw/bodies/RA_annotations/' # computerome
    img_dir = '/home/projects/ku_00017/data/raw/bodies/images_spanner' # computerome

    with open(f'{dict_dir}ra_ens_annotated_dict.pkl', 'rb') as file:
        attribute_dict = pickle.load(file)

    dataloaders = {}
    image_datasets = {}
    
    # it is now the same dataset just with different transformations
    image_datasets['train'] = CustomImageDataset(attribute_dict, attribute, img_dir, transform=data_transforms['train'])
    dataloaders['train'] = DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True)

    image_datasets['val'] = CustomImageDataset(attribute_dict, attribute, img_dir, transform=data_transforms['val']) # JUST TO SEE!!! 
    dataloaders['val'] = DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=True) #just set False...

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    return dataloaders, dataset_sizes 


def make(config, model_name):

    # Make the model
    weight_dict, model_dict = get_models()

    #Choose model and wieghts
    weights = weight_dict[model_name] # you need these later right as they hold the appropiate data transforamtions
    model = model_dict[model_name].to(device)
    # wandb.watch(model)

    # NOT re-train all parameters
    for param in list(model.parameters()):
        #param.requires_grad = True
        param.requires_grad = False


    # new model head for for retraining
    change_head(model_name, model, config['classes'])
    
    # Make the data
    #dataloaders, dataset_sizes, class_names = make_loader(batch_size=config.batch_size, weights = weights)
    dataloaders, dataset_sizes = make_loader(batch_size=config.batch_size,  weights = weights, attribute = config.attribute)
    
    # Make the loss and optimizer
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay = config.weight_decay, betas = config.betas)

    return model, criterion, optimizer, dataloaders, dataset_sizes #, class_names


def train_log(loss, example_ct, epoch):
    # Where the magic happens
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}.")


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


def train(model, model_name, train_loader, criterion, optimizer, config):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)

    # Run training and track with wandb
    example_ct = 0  # number of examples seen
    batch_ct = 0
    running_loss = 0.0
    train_RMSE_list = [] # only used in the last round

    for epoch in range(config.epochs):             

        ### trying this!
        if epoch == 1 and model_name != 'efficientnet_v2_s': # starting the 2nd epoch we do wnat to trian all 
                print('training all params!')
                for param in list(model.parameters()):
                    param.requires_grad = True
                   
        for _, (images, labels) in enumerate(train_loader):

            loss = train_batch(images, labels, model, optimizer, criterion)
        
            if epoch == config.epochs-1: # if it is the last round
                train_RMSE_list.append(torch.sqrt(loss).detach().cpu().numpy().item())

            example_ct +=  len(images)
            batch_ct += 1
            running_loss += loss

            # Report metrics every 20th batch - not running average right now
            if ((batch_ct + 1) % 100) == 0:
                #train_log(loss, example_ct, epoch) # this is the wand part
                train_log(running_loss/100, example_ct, epoch)
                running_loss = 0.0 # reset
    
    train_RMSE_array = np.array(train_RMSE_list)
    wandb.log({"train_rmse": train_RMSE_array.mean()})
    wandb.log({"train_rmse_dist": train_RMSE_array})

# this is inasmple now...
def test(model, test_loader): 
    model.eval()
    test_criterion = nn.MSELoss()

    with torch.no_grad():
        
        total = 0
        RMSE_list = []

        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            test_loss = test_criterion(outputs.squeeze(), labels)
            RMSE_list.append(torch.sqrt(test_loss).detach().cpu().numpy().item())

            total += labels.size(0) #so now you get the number of images - but not the number of mini batches. This is ok when you do not use it to norm.

        RMSE_array = np.array(RMSE_list)
        print(f"Average RMSE of the model on the {total} insample_val images: {RMSE_array.mean()}")
        wandb.log({"insample_val_rmse": RMSE_array.mean()})
        wandb.log({"insample_val_rmse_dist": RMSE_array})

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
      model, criterion, optimizer, dataloaders, dataset_sizes = make(config, model_name)
      print(model)

      # and use them to train the model
      train(model, model_name, dataloaders['train'], criterion, optimizer, config)

      # and test its final performance
      test(model, dataloaders['val'])

    return model


if __name__ == "__main__":

    wandb.login()

    input_dict1 = {'a': 'convnext_tiny',
                  'b' : 'efficientnet_v2_s',
                  'c' : 'regnet_x_8gf',
                  'd' : 'swin_t',
                  'e' : 'wide_resnet50_2',
                  'f' : 'squeezenet1_1',
                  'g' : 'shufflenet_v2_x0_5',
                  'h' : 'mnasnet0_5',
                  'i' : 'mobilenet_v3_small'}

    model_string = f"Choose model:\n "
    for k in input_dict1.keys():
        model_string += f"{k}) {input_dict1[k]}\n "

    print(model_string)

    input_string1 = input()
    if input_string1 in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']:
        model_name = input_dict1[input_string1]
        print(f'You choose {input_string1} : {model_name}')

    else:
        print('Wrong input')
        exit()

    input_dict2 = {'a' : 'all_negative_emotions_t1',
                   'b' : 'all_mass_protest',
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
    "weight_decay" : 0.01,
    'betas' : (0.9, 0.999),
    "classes" : 1,
    "epochs": 2, # 2 epochs is enough. Overfits efter 4-8
    "batch_size": 8 # 8 is good
    }

    # Build, train and analyze the model with the pipeline
    model = model_pipeline(hyperparameters)

    # save model
    done_model_dir = "/home/projects/ku_00017/people/simpol/scripts/bodies/Relative_attributes/Networks/Done_models/"
    path_SD = f'{done_model_dir}{model_name}_{attribute}_SD.pth'
    #path = f'{done_model_dir}{model_name}_{attribute}.pth'
    
    # do both; just in case.
    torch.save(model.state_dict(), path_SD)
    #torch.save(model, path)