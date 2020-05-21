import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from supplemental_code import detect_landmark
from pinhole_camera_model import generate_model_projection
from matching_model import Matcher
from visualizer import draw_pointcloud, draw_pointclouds

if __name__ == '__main__':
    image = './images/egill.jpg'
    ground_truth = torch.tensor(detect_landmark(image), dtype=torch.double)
    
    alpha = np.random.uniform(-1, 1, 20)
    delta = np.random.uniform(-1, 1, 30)

    w = [0.0, 0.0, 0.0]

    t = [0.0,0.0,0.0]

    f = 0
    n = 1

    lambda_alpha = 1
    lambda_delta = 1

    left = 0
    bottom = 0
    right = 1
    top = 1


    model = Matcher(f, n, left, bottom, right, top, alpha, delta, t, w)

    img, color = model.generate_image()