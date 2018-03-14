import time
import os
import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
import torchvision.datasets as dset
from torch.utils.data.sampler import SubsetRandomSampler

## SEED ##

seed = 57
torch.manual_seed(seed)

## CUDA
use_cuda = torch.cuda.is_available()

if use_cuda:
    torch.cuda.manual_seed(seed)
    kwargs = {'num_workers': 1, 'pin_memory': True}
else:
    kwargs = {}

# Models Path
retrain = False
path_model = './models/'
path_model_retrain = ""

# Training settings
root_data = './data'
exist_data = os.path.isdir(root_data)

if not exist_data:
    os.mkdir(root_data)

download = True if not exist_data else False

trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

train_set = dset.MNIST(root=root_data, train=True, transform=trans, download=download)
test_set = dset.MNIST(root=root_data, train=False, transform=trans)

valid_size = 0.1

num_train = len(train_set)
indices = list(range(num_train))
split = int(np.floor(valid_size * num_train))

train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

input_size = 28*28
output_size = 10
number_hidden_layers = 4
dict_weight_hidden_layers = [2500, 2000, 1500, 1000, 50]


exist_model = os.path.isdir(path_model)
if not exist_model:
  os.mkdir(path_model)

batch_size = 32
epochs = 50
lr = 0.01
momentum = 0.5


train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           sampler=train_sampler,
                                           **kwargs
                                           )

val_loader = torch.utils.data.DataLoader(dataset=test_set,
                                          batch_size=batch_size,
                                          sampler=valid_sampler,
                                          shuffle=False,
                                          **kwargs
                                          )

print("Total trainning batch number: {}".format(len(train_loader)))
print("Total validating batch number: {}".format(len(val_loader)))

loss_metric = nn.CrossEntropyLoss()

model = FFNN_Mnist(use_cuda = use_cuda, 
                   loss_metric = loss_metric, 
                   lr = lr, 
                   momentum = momentum, 
                   root_models = path_model, 
                   input_size = input_size, 
                   hidden_size_weight = dict_weight_hidden_layers, 
                   output_size = output_size, 
                   number_hidden_layers = number_hidden_layers,
                   verbose = False)


ti = time.clock()
if not retrain:
    model.train(epochs=epochs,
                train_loader=train_loader,
                val_loader=val_loader)
else:
    model.load_to_retrain(path_model_retrain)
    model.train(epochs=epochs,
                train_loader=train_loader,
                val_loader=val_loader)
    
tf = time.clock()

print("tiempo total = {}".format((tf-ti)/60))