import h5py
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from visualizer import draw_pointcloud

def compute_face(alpha, delta):
    mu_id = torch.tensor(fetch_weights('shape/model/mean'), dtype=torch.float64)
    mu_exp = torch.tensor(fetch_weights('expression/model/mean'), dtype=torch.float64)

    sigma_id = torch.tensor(np.sqrt(fetch_weights('shape/model/pcaVariance', False))[0:20], dtype=torch.float64)
    sigma_exp = torch.tensor(np.sqrt(fetch_weights('expression/model/pcaVariance', False))[0:30], dtype=torch.float64)

    E_id = fetch_weights('shape/model/pcaBasis', False)
    E_id = np.reshape(E_id, (-1, 3, 199))
    E_id = torch.tensor(E_id[:,:,0:20], dtype=torch.float64)

    E_exp = fetch_weights('expression/model/pcaBasis', False)
    E_exp = np.reshape(E_exp, (-1, 3, 100))
    E_exp = torch.tensor(E_exp[:,:,0:30], dtype=torch.float64)

    facial_pointcloud = mu_id + E_id @ (alpha*sigma_id) + E_exp @ (delta*sigma_exp)

    return facial_pointcloud

def fetch_weights(key: str, reshape: bool = True):
    bfm = h5py.File('./model2017-1_face12_nomouth.h5', 'r')
    # Select a specific weight from BFM
    weights = np.asarray(bfm[key], dtype=np.float32)
    # Sometimes you w i l l need t o r e s h a p e i t t o a p r ope r shape f o r
    # the pu rp o se o f t h i s a s si gnmen t
    if reshape:
        weights = np.reshape(weights,(-1,3))
    return weights

def generate_face_pointcloud(alpha = np.random.uniform(-1, 1, 20), delta = np.random.uniform(-1, 1, 30)):

    face_pointcloud = compute_face(alpha, delta)
    colors = fetch_weights('color/model/mean')

    return face_pointcloud, colors

if __name__ =='__main__':

    alpha = np.random.uniform(-1, 1, 20)
    delta = np.random.uniform(-1, 1, 30)

    face_pointcloud = compute_face(alpha, delta)
    colors = fetch_weights('color/model/mean')
    draw_pointcloud(face_pointcloud, colors)

