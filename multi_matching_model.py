import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

from morphable_model import generate_face_pointcloud
from pinhole_camera_model import generate_model_projection

class Matcher(nn.Module):
    """Some Information about Matcher"""
    def __init__(self, f, n, left, bottom, right, top, alpha, delta, t, w):
        super(Matcher, self).__init__()
        
        self.f = f
        self.n = n
        self.left = left
        self.bottom = bottom
        self.right = right
        self.top = top
        self.alpha = nn.Parameter(Variable(torch.tensor(alpha), requires_grad=True))

        self._init_weight_list(delta, t, w)

    def _init_weight_list(self, delta, t, w):
        deltaList = []
        tList = []
        wList = []

        for i in range(len(delta)):
            deltaList.append(nn.Parameter(Variable(torch.tensor(delta[i]), requires_grad=True)))
            wList.append(nn.Parameter(Variable(torch.tensor(w[i], dtype=torch.double), requires_grad=True)))
            tList.append(nn.Parameter(Variable(torch.tensor(t[i], dtype=torch.double), requires_grad=True)))

        self.delta = nn.ParameterList(deltaList)
        self.w = nn.ParameterList(wList)
        self.t = nn.ParameterList(tList)

    def forward(self, idx):

        train, colors = generate_model_projection(self.alpha, self.delta[idx], self.w[idx], self.f, self.n, self.left, self.bottom, self.right, self.top, self.t[idx], True)

        return train, colors

    def generate_image(self, idx, keypoints:bool):
        
        train, colors = generate_model_projection(self.alpha, self.delta[idx], self.w[idx], self.f, self.n, self.left, self.bottom, self.right, self.top, self.t[idx], keypoints)

        return train, colors

    def generate_3d_model(self, idx):
        points_3d, colors = generate_face_pointcloud(self.alpha, self.delta[idx])

        return points_3d, colors