import os
import numpy as np
import torch
import torchvision
from torchvision.utils import save_image
from os import listdir
from os.path import isfile, join

state = 'train'

def load_data(dataset_type, batch_size):
  # traindata_path = './Resources/frgc_chaudry/train'+dataset_type
  # valdata_path = './Resources/frgc_chaudry/val'+dataset_type
  path = './Resources/bosphorus_chaudhry/'+dataset_type

  if dataset_type == '_elev':
    mean_pix = [0.22587, 0.22587, 0.22587]
  elif dataset_type == '_depth':
    mean_pix = [0.2693, 0.2693, 0.2693]
  elif dataset_type == '_curv':
    mean_pix = [0.11014, 0.11014, 0.11014]

  dataset = torchvision.datasets.ImageFolder(
          root=path,
          transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=mean_pix,std=[1,1,1])
          ])
      )
  #print("Train: Detected Classes are: ", train_dataset.class_to_idx)
  dataset_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=0,
    shuffle=False
  )
  return dataset_loader



dataset_type1 = '_elev'
dataset_loader1 = load_data(dataset_type1,1)

dataset_type2 = '_curv'
dataset_loader2 = load_data(dataset_type2,1)

filesName = [f for f in listdir('./Resources/bosphorus_chaudhry/'+ dataset_type1+'/1/') if isfile(join('./Resources/bosphorus_chaudhry/'+ dataset_type1+'/1/' , f))]
filesName.sort()

first_dataset = []
second_dataset = []
final_dataset = []
target_dataset = []

for i,(image,target) in enumerate(dataset_loader1):
  image = image.narrow(1,0,1)
  first_dataset.append(image)
  #target_dataset.append(target)
  if ((i%5000)== 0):
    print(i/len(dataset_loader1))

for i,(image,target) in enumerate(dataset_loader2):
  image = image.narrow(1,0,1)
  second_dataset.append(image)
  if ((i % 5000) == 0):
    print(i/len(dataset_loader2))

for i in np.arange(len(first_dataset)):
  final_dataset.append(torch.cat((torch.cat((first_dataset[i],second_dataset[i]),1),torch.zeros(65536).reshape([1,1,256,256])),1))
  path = './Resources/bosphorus_chaudhry/special/'+dataset_type1+'_'+dataset_type2+'/'
  if not os.path.exists(path):
    os.makedirs(path)
  save_image(final_dataset[i], path+filesName[i])
  print(i/len(first_dataset))
