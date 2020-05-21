import numpy as np
from morphable_model import generate_face_pointcloud
from visualizer import draw_pointcloud

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

def compute_viewport(v_l, v_b, v_r, v_t):
    viewport_matrix = np.array([
        [(v_r-v_l)/2, 0, 0 , (v_r+v_l)/2],
        [0, (v_t-v_b)/2, 0, (v_t+v_b)/2],
        [0, 0, 0.5, 0.5],
        [0, 0, 0, 1]
    ])
    return viewport_matrix

def compute_projection(l, b, r, t, f, n):
    projection_matrix = np.array([
        [2*n/(r-l), 0, (r+l)/(r-l), 0],
        [0, 2*n/(t-b), (t+b)/(t-b), 0],
        [0, 0, -(f+n)/(f-n), -(2*f*n)/(f-n)],
        [0, 0, -1, 0]
    ])

    return projection_matrix

def compute_R(w):
    x_deg = (w[0] * np.pi/180).reshape(1,1)
    R_top = torch.cat((torch.cos(x_deg), -torch.sin(x_deg)), dim=1)
    R_bottom = torch.cat((torch.sin(x_deg), torch.cos(x_deg)), dim=1)
    R_square = torch.cat((R_top, R_bottom), dim=0)
    R_x_top = torch.tensor([0.0,0.0],dtype=torch.double, requires_grad=True).reshape(1,2)
    R_x_left = torch.tensor([1.0,0.0,0.0],dtype=torch.double, requires_grad=True).reshape(3,1)
    R_x = torch.cat((R_x_top, R_square), dim=0)
    R_x = torch.cat((R_x_left, R_x), dim=1)
    

    
    y_deg = (w[1] * np.pi/180).reshape(1,1)
    top_zero = torch.tensor([0], dtype=torch.double, requires_grad=True).reshape(1,1)
    R_y_top = torch.cat( (torch.cos(y_deg), top_zero), dim=1 )
    R_y_top = torch.cat((R_y_top, torch.sin(y_deg)), dim=1)
    middle_row = torch.tensor([0,1,0], requires_grad=True, dtype=torch.double).reshape(1,3)
    R_y = torch.cat((R_y_top, middle_row), dim=0)
    bottom_zero = torch.tensor([0], dtype=torch.double, requires_grad=True).reshape(1,1)
    R_y_bottom = torch.cat((-torch.sin(y_deg), bottom_zero), dim=1)
    R_y_bottom = torch.cat((R_y_bottom, torch.cos(y_deg)), dim=1)
    R_y = torch.cat((R_y, R_y_bottom), dim=0)

    
    z_deg = (w[2] * np.pi/180).reshape(1,1)
    z_top = torch.cat( (torch.cos(z_deg), -torch.sin(z_deg) ), dim=1)
    z_mid = torch.cat( (torch.sin(z_deg), torch.cos(z_deg) ), dim=1)
    z_bottom = torch.tensor([0,0], dtype=torch.double, requires_grad=True).reshape(1,2)
    z_right = torch.tensor([0,0,1], dtype=torch.double, requires_grad=True).reshape(3,1)
    R_z = torch.cat( (z_top, z_mid), dim=0)
    R_z = torch.cat( (R_z, z_bottom), dim=0)
    R_z = torch.cat( (R_z, z_right), dim=1)

    R = R_z @ R_y @ R_x

    return R

def compute_T(t, w):
    R = compute_R(w)
    T = torch.cat((R, t.reshape((len(t), 1))), 1)
    bottom_row = torch.tensor([0.0,0.0,0.0,1.0], dtype=torch.double).reshape((1, 4))
    T = torch.cat((T, bottom_row))
    return T

def generate_model_projection(alpha, delta, w, f, n, left, bottom, right, top, t, only_keypoints = False):

    V = compute_viewport(left, bottom, right, top)
    P = compute_projection(left, bottom, right, top, f, n)

    pi = V @ P

    T = compute_T(t, w)

    points, colors = generate_face_pointcloud(alpha, delta)

    extra_dim = torch.ones(len(points), dtype=torch.double).reshape(len(points), 1)

    points = torch.cat((points, extra_dim), dim=1)

    result = (torch.tensor(pi) @ T.type(torch.double)  @ points.t()).t()

    if only_keypoints:
        indicies = np.loadtxt('model2017-1_face12_nomouth.anl', dtype=int).tolist()
        result = result[torch.tensor(indicies)]
        colors = colors[torch.tensor(indicies)]

    return result, colors

if __name__ == '__main__':
    theta_x = 0
    theta_y = 0
    theta_z = 0

    f = 0
    n = 1

    left = 0
    bottom = 0
    right = 1
    top = 1

    t_1 = 0
    t_2 = 0
    t_3 = 0


    draw_pointcloud(result[:, 0:3], color)
    #plt.scatter(points[:, 0], points[:,1], color)
    #plt.show()