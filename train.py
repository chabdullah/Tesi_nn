import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix
from time import time

from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from NnModel import NnModel
from itertools import product

#torch.set_printoptions(threshold=10000000)

n_epochs = 80
parameters = dict(
    lr = [.01]
    , batch_size=[16]
    , datasets=['_elev', '_depth', '_curv']
    , dim_descrittore=[1024]
    , kernel_size=[5]
)
param_values = [v for v in parameters.values()]
momentum = 0.5
log_interval = 10
device = 'cuda:1'


def load_data(batch_size):
  traindata_path = './Resources/frgc_chaudry/train'+dataset_type
  valdata_path = './Resources/frgc_chaudry/val'+dataset_type

  if dataset_type == '_elev':
    mean_pix = [0.22587, 0.22587, 0.22587]
  elif dataset_type == '_depth':
    mean_pix = [0.2693, 0.2693, 0.2693]
  elif dataset_type == '_curv':
    mean_pix = [0.11014, 0.11014, 0.11014]

  train_dataset = torchvision.datasets.ImageFolder(
          root=traindata_path,
          transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(64),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=mean_pix,std=[1,1,1])
          ])
      )
  val_dataset = torchvision.datasets.ImageFolder(
    root=valdata_path,
    transform=torchvision.transforms.Compose([
      torchvision.transforms.Resize(64),
      torchvision.transforms.RandomHorizontalFlip(),
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize(mean=mean_pix,std=[1,1,1])
    ])
  )
  #print("Train: Detected Classes are: ", train_dataset.class_to_idx)
  train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=0,
    shuffle=True
  )
  val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    num_workers=0,
    shuffle=True
  )
  return train_dataset, val_dataset, train_loader, val_loader




def train(epoch):
  network.train()
  total_loss = 0
  correct = 0
  for batch_id, (data, target) in enumerate(train_loader):
    data = data.narrow(1,0,1)
    data = data.to(device)
    target = target.to(device)
    optimizer.zero_grad()
    output = network(data)
    loss = criterion(output, target)
    total_loss += loss.item()
    loss.backward()
    optimizer.step()
    pred = output.data.max(1, keepdim=True)[1]
    correct += pred.eq(target.data.view_as(pred)).sum()
    if batch_id % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_id * len(data), len(train_loader.dataset),
        100. * batch_id / len(train_loader), loss.item()))
      torch.save(network.state_dict(), './savedState/model'+dataset_type+'.pth')
      torch.save(optimizer.state_dict(), './savedState/optimizer'+dataset_type+'.pth')
  tb.add_scalar(dataset_type+' Train Loss', 100.0 * total_loss/len(train_loader.dataset), epoch)
  tb.add_scalar(dataset_type + ' Train Accuracy', 100 * correct / len(train_loader.dataset), epoch)


  for name, param in network.named_parameters():
    tb.add_histogram(name, param, epoch)
    #tb.add_histogram(f'{name}.grad', param.grad, epoch)



def test(epoch):
  network.eval()
  train_loss = 0
  correct = 0
  with torch.no_grad():
    for batch_id, (data, target) in enumerate(val_loader):
      data = data.narrow(1,0,1)
      data = data.to(device)
      target = target.to(device)
      output = network(data)
      train_loss += criterion(output, target).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  train_loss /= len(val_loader.dataset)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    train_loss, correct, len(val_loader.dataset),
    100. * correct / len(val_loader.dataset)))
  tb.add_scalar(dataset_type + ' Test Loss', 100.0 * train_loss, epoch)
  tb.add_scalar(dataset_type + ' Test Accuracy', 100. * correct/len(val_loader.dataset), epoch)







for lr, batch_size, dataset_type, dim_descrittore, kernel_size in product(*param_values):
  torch.cuda.empty_cache()

  train_dataset, val_dataset, train_loader, val_loader = load_data(batch_size)
  network = NnModel(dim_descrittore, kernel_size)
  #network.load_state_dict(torch.load('./savedState/model' + dataset_type + '.pth'))
  network = network.to(device)
  optimizer = optim.SGD(network.parameters(), lr=lr, momentum=momentum)
  #optimizer.load_state_dict(torch.load('./savedState/optimizer' + dataset_type + '.pth'))
  scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
  criterion = nn.CrossEntropyLoss()

  images, labels = next(iter(train_loader))
  grid = torchvision.utils.make_grid(images)

  tb = SummaryWriter(comment=f' batch_size={batch_size} dataset={dataset_type} dim_descrittore={dim_descrittore} kernel_size= {kernel_size}')
  tb.add_image('Faces', grid)


  print('\033[92m' + 'batch_size=%s dataset=%s \ndim_descrittore=%s kernel_size=%s \033[0m'% (batch_size, dataset_type, dim_descrittore, kernel_size))
  for epoch in range(n_epochs):
    scheduler.step()
    train(epoch)
    test(epoch)
  tb.close()



# watch nvidia-smi
# source /home/cferrari/python/virtualEnvs/activeface-torch/bin/activate