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
        self.delta = nn.Parameter(Variable(torch.tensor(delta), requires_grad=True))

        self.w = nn.Parameter(Variable(torch.tensor(w, dtype=torch.double), requires_grad=True))

        self.t = nn.Parameter(Variable(torch.tensor(t, dtype=torch.double), requires_grad=True))


    def forward(self):

        train, colors = generate_model_projection(self.alpha, self.delta, self.w, self.f, self.n, self.left, self.bottom, self.right, self.top, self.t, True)

        return train, colors

    def generate_image(self, keypoints:bool):
        
        train, colors = generate_model_projection(self.alpha, self.delta, self.w, self.f, self.n, self.left, self.bottom, self.right, self.top, self.t, keypoints)

        return train, colors