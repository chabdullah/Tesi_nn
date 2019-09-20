import time
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from NnModelFeature import NnModelFeature
import torch
from torchvision import transforms
import torch.nn as nn
from os.path import join
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
import pickle
from scipy.spatial.distance import cdist
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from itertools import product
import matplotlib.pylab as plt


parameters = dict(
      datasets=['_curv', '_depth', '_elev']
    , pca = [True, False]
    , proto=['NvsN', 'NvsE', 'NvsA']
)
param_values = [v for v in parameters.values()]


def autolabel(ax,rects):
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * h, '%.1f' % float(h),
                ha='center', va='bottom')


def plotResults():
    dataset = ['_curv', '_depth', '_elev']

    fig, ax = plt.subplots(2, 1, constrained_layout=False)
    for i,pca in enumerate(['PCA_on','PCA_off']):
        NvsN_rank = []
        NvsE_rank = []
        NvsA_rank = []
        for ds in ['_curv', '_depth', '_elev']:
            NvsN_rank.append(results[ds][pca]['NvsN'])
            NvsE_rank.append(results[ds][pca]['NvsE'])
            NvsA_rank.append(results[ds][pca]['NvsA'])
        NvsN_rank = list(map(float, NvsN_rank))
        NvsE_rank = list(map(float, NvsE_rank))
        NvsA_rank = list(map(float, NvsA_rank))

        ind = np.arange(len(dataset))
        width = 0.2
        rects1 = ax[i].bar(ind, NvsN_rank, width, color='r')
        rects2 = ax[i].bar(ind + width, NvsE_rank, width, color='g')
        rects3 = ax[i].bar(ind + width * 2, NvsA_rank, width, color='b')
        ax[i].set_xticks(ind + width)
        ax[i].set_xticklabels(list(dataset))
        ax[i].legend((rects1[0], rects2[0], rects3[0]), ('NvsN', 'NvsE', 'NvsA'))
        ax[i].set_title(pca)
        ax[i].set_ylabel('Rank value')

        autolabel(ax[i],rects1)
        autolabel(ax[i],rects2)
        autolabel(ax[i],rects3)
    plt.show()


def extract_feat_bosphorus(images_path, dataset_type, pretrained):

    # Create model
    device = "cpu"
    model = NnModelFeature(1024,5)

    if pretrained:
        weights_path = './savedState/model' + dataset_type + '.pth'
        state_dict = torch.load(weights_path,map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
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

    features_all = np.zeros((len(images), 1024), dtype=float)

    for i, imfile in enumerate(images):
        im = cv2.imread(join(images_path, imfile))
        im = cv2.resize(im, (64, 64))
        im = torch.Tensor(im).view(1, 3, 64, 64)

        im -= torch.Tensor(np.array(mean_pix)).view(1, 3, 1, 1)
        im = im.to(device)

        feat = model(im)

        features_all[i, :] = feat.cpu().detach().numpy()
        #print(i)

    return features_all, images


def matching(dataset_type='_curv', proto='NvsN', pca_on=False):

    with open('./savedState/featuresBosphorus/' + dataset_type + '.pkl', 'rb') as f:
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

        # Standardize gallery    (-mean diviso varianza)
        # Istanzia pca
        # pca.fit(gallery)
        # ...e poi transform sia gallery che probe.
        # Infine matching
        # gallery_feat = (gallery_feat - np.mean(gallery_feat)) / np.var(gallery_feat)


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
    if pca_on:
        scaler = StandardScaler()
        scaler.fit(gallery_feat)
        gallery_feat = scaler.transform(gallery_feat)
        probe_feat = scaler.transform(probe_feat)

        # Make an instance of the Model
        pca = PCA(.95)
        pca.fit(gallery_feat)

        gallery_feat = pca.transform(gallery_feat)
        probe_feat = pca.transform(probe_feat)

    gallery_feat = np.array(gallery_feat)
    probe_feat = np.array(probe_feat)
    gallery_id = np.array(gallery_id)
    probe_id = np.array(probe_id)


    dists = cdist(probe_feat, gallery_feat, 'cosine')
    dists = -dists + 1
    rank1 = 0
    print()

    for i in range(dists.shape[0]):
        idx = np.argmax(dists[i, :])
        if probe_id[i] == gallery_id[idx]:
        #if probe_id[i] == gallery_id[random.randrange(0,gallery_id.size,1)]:
            rank1 += 1
    rank1 = (rank1 / dists.shape[0]) * 100
    return (str(rank1))





# ATTENZIONE: le immagini 3 canali sono RGB, ma la rete è addestrata BGR. Se carico con cv2, il formato
# dovrebbe essere già apposto poerché carica in BGR

results = {'_depth': {'PCA_on':{'NvsN':0,'NvsE':0,'NvsA':0}, 'PCA_off':{'NvsN':0,'NvsE':0,'NvsA':0}}, '_curv': {'PCA_on':{'NvsN':0,'NvsE':0,'NvsA':0}, 'PCA_off':{'NvsN':0,'NvsE':0,'NvsA':0}}, '_elev': {'PCA_on':{'NvsN':0,'NvsE':0,'NvsA':0}, 'PCA_off':{'NvsN':0,'NvsE':0,'NvsA':0}}}
for dataset_type, pca_on, proto in product(*param_values):
    for feat_extr in [ False]:
        if feat_extr:
            features_all, images = extract_feat_bosphorus(images_path='./Resources/bosphorus_chaudhry/' + dataset_type, dataset_type=dataset_type,
                                                          pretrained=True)
            print("done")
            with open('./savedState/featuresBosphorus/' + dataset_type + '.pkl', 'wb') as f:
                pickle.dump([features_all, images], f)
        else:
            rank1 = matching(dataset_type=dataset_type, proto=proto, pca_on=pca_on)
            if pca_on:
                results[dataset_type]['PCA_on'][proto] = rank1
            else:
                results[dataset_type]['PCA_off'][proto] = rank1

plotResults()
