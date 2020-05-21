import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dlib
import matplotlib.pyplot as plt

from supplemental_code import detect_landmark
from pinhole_camera_model import generate_model_projection
from matching_model import Matcher
from visualizer import draw_pointcloud, draw_pointclouds

def format_image(img):

    points = []
    colors = []

    for i in range(len(img[0])):
        for j in range(len(img)):
            points.append((i,j, -1))
            colors.append(img[j][i])

    return np.array(points), np.array(colors)/255

if __name__ == '__main__':
    image = './images/egill.jpg'
    image = dlib.load_rgb_image(image)

    img_points, img_color = format_image(image)

    ground_truth = torch.tensor(detect_landmark(image), dtype=torch.double)

    #draw_pointcloud(img_points, img_color)

    alpha = np.random.uniform(-1, 1, 20)
    delta = np.random.uniform(-1, 1, 30)

    w = [0.0, 0, 180.0]

    t = [0,image.shape[1]/2, -500.0]

    f = 0
    n = 2

    lambda_alpha = 1
    lambda_delta = 1

    left = 0
    bottom = 0
    right = 1080
    top = 1920


    model = Matcher(f, n, left, bottom, right, top, alpha, delta, t, w)
    optimizer = optim.Adam(model.parameters(), lr=0.2)

    loss = torch.tensor(np.inf)

    i = 0

    while loss > 100:
        optimizer.zero_grad()

        results, colors = model.forward()

        results = results[:,0:2]

        lan_loss = torch.sum((torch.sqrt(torch.sum((results - ground_truth) ** 2, dim=1)) ** 2), dim=0)/68

        reg_loss = lambda_alpha * torch.sum(model.alpha**2) + lambda_delta * torch.sum(model.delta ** 2)

        loss = lan_loss + reg_loss

        print(loss.item())

        loss.backward()
        optimizer.step()
        

        if i%1 == 0: # and loss.item() < 500:
            extra_dim = torch.ones(len(ground_truth), dtype=torch.double).reshape(len(ground_truth), 1)
            ground_truth_3d = torch.cat((ground_truth, extra_dim), dim=1)
            fitted_image, colors = model.generate_image(False)
            #draw_pointclouds(fitted_image.detach().numpy()[:, 0:3], img_points, colors, img_color)

            plt.imshow(img_points[:, 0:2])
            plt.show()
        i += 1








