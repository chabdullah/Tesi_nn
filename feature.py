import time
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from NnModelFeature import NnModel
import torch
from torchvision import transforms
import torch.nn as nn
from tensorboardX import SummaryWriter
from os.path import join
from os import listdir
from os.path import isfile, join
#import cv2
import numpy as np
import pickle
from scipy.spatial.distance import cdist


def extract_feat_bosphorus(images_path='./Resources/bosphorus_chaudhry/depth', dataset_type='_curv', ep='19', pretrained=False):

    # Create VGG pytorch model
    device = "cuda:0"
    model = NnModel()

    if pretrained:
        # weights_path = 'pretrained/VGG_FACE.pth'
        # state_dict = torch.load(weights_path)
        # model.load_state_dict(state_dict)
        model.load_state_dict(torch.load('./savedState/model' + dataset_type + '.pth'))
    else:
        weights_path = 'states/epoch_' + ep + '_' + dataset_type + '.pth.tar'
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict["state_dict"])

    # Get mean pix values
    if dataset_type == '_elev':
        mean_pix = [0.22587, 0.22587, 0.22587]
    elif dataset_type == '_depth':
        mean_pix = [0.2693, 0.2693, 0.2693]
    elif dataset_type == '_curv':
        mean_pix = [0.11014, 0.11014, 0.11014]

    # Create model

    model = model.to(device)
    model.eval()

    # Loop over images
    images = [f for f in listdir(images_path) if isfile(join(images_path, f))]

    features_all = np.zeros((len(images), 1000), dtype=float)

    for i, imfile in enumerate(images):
        im = cv2.imread(join(images_path, imfile))
        im = cv2.resize(im, (64, 64))
        im = torch.Tensor(im).view(1, 1, 64, 64)

        im -= torch.Tensor(np.array(mean_pix)).view(1, 1, 1, 1)
        im = im.to(device)

        feat = model(im)

        features_all[i, :] = feat.cpu().detach().numpy()
        print(i)

    return features_all, images


def matching(dataset_type='_curv', proto='NvsN'):

    with open('./Resources/bosphorus_chaudhry/' + dataset_type + '.pkl', 'rb') as f:
        features_all, images = pickle.load(f)

    # Normalize features
    # for i in range(len(images)):
    #     feat = features_all[i, :]
    #     if sum(feat) == 0:
    #         print(i)
    #     features_all[i, :] = feat / np.linalg.norm(feat, ord=2)
    # print("Normalization Done")

    gallery_feat = []
    probe_feat = []
    gallery_id = []
    probe_id = []

    # Build Gallery and probe
    # Remove occluded faces and rotations
    for i, (f, name) in enumerate(zip(features_all, images)):

        # Gallery
        if 'N_N_0' in name:
            gallery_feat.append(f)
            gallery_id.append(int(name[2:name.find('_')]))

        # Probe depending on protocol
        if proto == 'NvsN':
            if ('N_N_1' in name) or ('N_N_2' in name):
                probe_feat.append(f)
                probe_id.append(int(name[2:name.find('_')]))
        elif proto == 'NvsE':
            if ('N_N' not in name) or ('_O_' not in name):
                probe_feat.append(f)
                probe_id.append(int(name[2:name.find('_')]))
        elif proto == 'NvsA':
            if ('_O_' not in name):
                probe_feat.append(f)
                probe_id.append(int(name[2:name.find('_')]))
    print('Gallery and probe sets built!')

    gallery_feat = np.array(gallery_feat)
    probe_feat = np.array(probe_feat)
    gallery_id = np.array(gallery_id)
    probe_id = np.array(probe_id)

    dists = cdist(probe_feat, gallery_feat, 'cosine')
    dists = -dists + 1
    rank1 = 0

    for i in range(dists.shape[0]):
        idx = np.argmax(dists[i, :])
        if probe_id[i] == gallery_id[idx]:
            rank1 += 1
    rank1 = rank1 / dists.shape[0]
    print(str(rank1) + '-epoch-' + str(ep))


#if __name__ == "__main__":

    # ATTENZIONE: le immagini 3 canali sono RGB, ma la rete è addestrata BGR. Se carico con cv2, il formato
    # dovrebbe essere già apposto poerché carica in BGR

    # feat_extr = True
    # ep = '19'
    # dataset_type = '_curv'
    # proto = 'NvsN'
    # if feat_extr:
    #     features_all, images = extract_feat_bosphorus(images_path='./Resources/bosphorus_chaudhry/' + dataset_type, dataset_type=dataset_type, ep=ep,
    #                                                   pretrained=True)
    #     print("done")
    #     with open('/home/abdullah/Scrivania/Tesi/NeuralNetwork/savedState/featuresBosphorus' + dataset_type + '.pkl', 'wb') as f:
    #         pickle.dump([features_all, images], f)
    # else:
    #     matching(dataset_type=dataset_type, proto=proto)

#images_path = './Resources/bosphorus_chaudhry/_elev'
images_path = './Resources//frgc_chaudry/train_curv/02463'
images = [f for f in listdir(images_path) if isfile(join(images_path, f))]
name = images[0]
print(name)
print(name[2:name.find('_')])
