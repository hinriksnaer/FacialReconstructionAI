import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dlib
import matplotlib.pyplot as plt

from supplemental_code import detect_landmark, render
from pinhole_camera_model import generate_model_projection
from matching_model import Matcher
from visualizer import draw_pointcloud, draw_pointclouds
from morphable_model import fetch_weights
from texturing import compute_colors, texture_image, format_image

if __name__ == '__main__':
    image = './images/aronset/aron3.jpg'
    image = dlib.load_rgb_image(image)

    img_points, img_color = format_image(image)

    ground_truth = torch.tensor(detect_landmark(image), dtype=torch.double, device='cuda')

    #draw_pointcloud(torch.cat((ground_truth, torch.ones((len(ground_truth),1), dtype=torch.double, device='cuda')), dim=1).detach().cpu().numpy())

    alpha = np.random.uniform(-1, 1, 20)
    delta = np.random.uniform(-1, 1, 30)

    w = [0.0, 180.0, 180.0]

    t = [image.shape[1]/5,image.shape[0]/5,-500]#[image.shape[0]/2,image.shape[1]/2, -500.0]

    f = 0
    n = 1500

    lambda_alpha = 0.5
    lambda_delta = 1

    left = 0
    bottom = 0
    right = image.shape[1]
    top = image.shape[0]

    early_stopping = 80

    model = Matcher(f, n, left, bottom, right, top, alpha, delta, t, w).to('cuda')
    optimizer = optim.Adam(model.parameters(), lr=0.2)

    loss = torch.tensor(np.inf, device='cuda')

    i = 0

    loss_list = []

    while loss > 1:
        optimizer.zero_grad()

        results, colors = model.forward()

        results_uv = results[:,0:2]

        lan_loss = torch.sum((torch.sqrt(torch.sum((results_uv - ground_truth) ** 2, dim=1)) ** 2), dim=0)/68

        reg_loss = lambda_alpha * torch.sum(model.alpha**2) + lambda_delta * torch.sum(model.delta ** 2)

        loss = lan_loss + reg_loss

        print(loss.item())
        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()
        

        if loss.item() < early_stopping:
            extra_dim = torch.ones(len(ground_truth), dtype=torch.double, device='cuda').reshape(len(ground_truth), 1)
            ground_truth_3d = torch.cat((ground_truth, extra_dim), dim=1)
            fitted_image, colors = model.generate_image(False)

            z_dim = torch.ones((len(fitted_image), 1), device='cuda', dtype=torch.double)
            draw_pointclouds(torch.cat((fitted_image[:,0:2], z_dim), dim=1).detach().cpu().numpy(), img_points, colors, img_color)

            ground_truth_z = torch.ones((len(ground_truth), 1), device='cuda', dtype=torch.double)

            draw_pointclouds(torch.cat((ground_truth, ground_truth_z), dim=1).detach().cpu().numpy(), torch.cat((results_uv, ground_truth_z), dim=1).detach().cpu().numpy())

            points_3d, colors_3d = model.generate_3d_model()
            draw_pointcloud(points_3d.detach().cpu().numpy(), colors_3d)

            if loss.item() < early_stopping:
                estimated_colors = compute_colors(fitted_image.detach().cpu().numpy(), img_points, img_color)
                draw_pointcloud(fitted_image.detach().cpu().numpy()[:, 0:3], estimated_colors)
                draw_pointcloud(points_3d.detach().cpu().numpy()[:, 0:3], estimated_colors)
                texture_image(fitted_image.detach().cpu().numpy()[:, 0:3], estimated_colors, image.shape[0], image.shape[1])

            x = [i+1 for i in range(len(loss_list))]
            
            plt.plot(x, loss_list, label='Loss')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()

            plt.savefig('./losses.png')

            plt.imshow(img_points[:, 0:2])
            #plt.show()


        i += 1








