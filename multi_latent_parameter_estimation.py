import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dlib
import matplotlib.pyplot as plt

from supplemental_code import detect_landmark, render
from pinhole_camera_model import generate_model_projection
from multi_matching_model import Matcher
from visualizer import draw_pointcloud, draw_pointclouds
from morphable_model import fetch_weights
from texturing import compute_colors, texture_image, format_image

if __name__ == '__main__':

    image_path = [
        './images/aronset/aron1.jpg',
        './images/aronset/aron2.jpg',
        './images/aronset/aron3.jpg',
        './images/aronset/aron5.jpg',
        './images/aronset/aron6.jpg'
        ]
    '''
    './images/hinrikset/hinrik1.jpg',
    './images/hinrikset/hinrik2.jpg',
    './images/hinrikset/hinrik3.jpg',
    './images/hinrikset/hinrik4.jpg',
    './images/hinrikset/hinrik5.jpg',
    './images/hinrikset/hinrik6.jpg',
    './images/hinrikset/hinrik7.jpg'
    '''
    n_img = len(image_path)
    img_points = []
    img_color = []
    ground_truth = []
    delta = []
    w = []
    t = []

    threshold = 500
    #t = [[189,235,-475], [231,247,-463], [192,229,-491], [193,242,-468], [175,253,-466], [177,238,-480], [224, 262, -446]]
    for img_path in image_path:

        img = dlib.load_rgb_image(img_path)
        image_points, image_color = format_image(img)
        img_points.append(image_points)
        img_color.append(image_color)
        ground_truth.append(torch.tensor(detect_landmark(img), dtype=torch.double, device='cuda'))

        delta.append(np.random.uniform(-1, 1, 30))

        w.append([0.0, 180.0, 180.0])
        t.append([img.shape[1]/5,img.shape[0]/5,-500])

    alpha = np.random.uniform(-1, 1, 20)

    f = 0
    n = 1500

    lambda_alpha = 0.5
    lambda_delta = 1

    right = 0
    top = 0
    left = img.shape[1]
    bottom = img.shape[0]


    model = Matcher(f, n, left, bottom, right, top, alpha, delta, t, w).to('cuda')
    optimizer = optim.Adam(model.parameters(), lr=0.2)

    loss = torch.tensor(np.inf, device='cuda')

    i = 0

    loss_list = []

    losses_list = [[],[],[],[],[]]
    while loss > 1:

        print('loss: ', loss_list)
        loss_list = []


        for k in range(n_img):
            optimizer.zero_grad()

            results, colors = model.forward(k%n_img)

            results_uv = results[:,0:2]

            lan_loss = torch.sum((torch.sqrt(torch.sum((results_uv - ground_truth[k%n_img]) ** 2, dim=1)) ** 2), dim=0)/68

            reg_loss = lambda_alpha * torch.sum(model.alpha**2) + lambda_delta * torch.sum(model.delta[k%n_img] ** 2)

            loss = lan_loss + reg_loss

            loss_list.append(loss.item())
            losses_list[k].append(loss.item())
            loss.backward()
            optimizer.step()

        for j in range(len(image_path)):
            if len(loss_list) == len(image_path) and np.sum(loss_list) < threshold:
                extra_dim = torch.ones(len(ground_truth[j%n_img]), dtype=torch.double, device='cuda').reshape(len(ground_truth[j%n_img]), 1)
                ground_truth_3d = torch.cat((ground_truth[j%n_img], extra_dim), dim=1)
                fitted_image, colors = model.generate_image(j%n_img, False)
                draw_pointclouds(fitted_image.detach().cpu().numpy()[:, 0:3], img_points[j%n_img], colors, img_color[j%n_img])

                points_3d, colors_3d = model.generate_3d_model(j%n_img)
                draw_pointcloud(points_3d.detach().cpu().numpy(), colors_3d)

                estimated_colors = compute_colors(fitted_image.detach().cpu().numpy(), img_points[j%n_img], img_color[j%n_img])
                draw_pointcloud(fitted_image.detach().cpu().numpy()[:, 0:3], estimated_colors)
                texture_image(fitted_image.detach().cpu().numpy()[:, 0:3], estimated_colors, img.shape[0], img.shape[1])

                x = [i+1 for i in range(len(losses_list[0]))]
                if j == len(image_path)-1:
                    for z, losslist in enumerate(losses_list):
                        plt.plot(x, losslist, label='Image '+str(z+1))
                    plt.xlabel('Iteration')
                    plt.ylabel('Loss')
                    plt.legend()
                    plt.show()


        i += 1






