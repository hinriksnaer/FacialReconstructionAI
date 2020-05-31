import numpy as np
import h5py

from morphable_model import fetch_weights
from supplemental_code import render
from visualizer import draw_pointcloud, draw_pointclouds

def format_image(img, normalize=True):
    points = []
    colors = []

    for i in range(len(img[0])):
        for j in range(len(img)):
            points.append((i,j, -1))
            colors.append(img[j][i])
    if normalize == False:
        return np.array(points), np.array(colors)
    return np.array(points), np.array(colors)/255

def bilinear_interpolation(x, y, points):

    points = sorted(points)  # order points by x, then by y
    (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

    return (q11 * (x2 - x) * (y2 - y) +
            q21 * (x - x1) * (y2 - y) +
            q12 * (x2 - x) * (y - y1) +
            q22 * (x - x1) * (y - y1)
           ) / ((x2 - x1) * (y2 - y1) + 0.0)

def compute_colors(projection, image, image_colors):
    colors = []

    for points in projection:
        lower_left_point = np.where((image[:, 0] == np.floor(points[0])) & (image[:, 1] == np.floor(points[1])))
        lower_right_point = np.where((image[:, 0] == np.ceil(points[0])) & (image[:, 1] == np.floor(points[1])))
        top_left_point = np.where((image[:, 0] == np.floor(points[0])) & (image[:, 1] == np.ceil(points[1])))
        top_right_point = np.where((image[:, 0] == np.ceil(points[0])) & (image[:, 1] == np.ceil(points[1])))
    
        if len(lower_left_point) != 1 or len(lower_right_point) != 1 or len(top_left_point) != 1 or len(top_right_point) !=1 :
            raise ValueError('POINT IS MISSING')

        ground_truth_points = [
            (np.floor(points[0]), np.floor(points[1]), image_colors[lower_left_point, 0][0][0]),
            (np.ceil(points[0]), np.floor(points[1]), image_colors[lower_right_point, 0][0][0]),
            (np.floor(points[0]), np.ceil(points[1]), image_colors[top_left_point, 0][0][0]),
            (np.ceil(points[0]), np.ceil(points[1]), image_colors[top_right_point, 0][0][0])
        ]

        r = bilinear_interpolation(points[0], points[1], ground_truth_points)

        ground_truth_points = [
            (np.floor(points[0]), np.floor(points[1]), image_colors[lower_left_point, 1][0][0]),
            (np.ceil(points[0]), np.floor(points[1]), image_colors[lower_right_point, 1][0][0]),
            (np.floor(points[0]), np.ceil(points[1]), image_colors[top_left_point, 1][0][0]),
            (np.ceil(points[0]), np.ceil(points[1]), image_colors[top_right_point, 1][0][0])
        ]

        g = bilinear_interpolation(points[0], points[1], ground_truth_points)

        ground_truth_points = [
            (np.floor(points[0]), np.floor(points[1]), image_colors[lower_left_point, 2][0][0]),
            (np.ceil(points[0]), np.floor(points[1]), image_colors[lower_right_point, 2][0][0]),
            (np.floor(points[0]), np.ceil(points[1]), image_colors[top_left_point, 2][0][0]),
            (np.ceil(points[0]), np.ceil(points[1]), image_colors[top_right_point, 2][0][0])
        ]

        b = bilinear_interpolation(points[0], points[1], ground_truth_points)

        colors.append((r,g,b))

    return np.array(colors)

def texture_image(projection, image_colors, width, height):
    bfm = h5py.File('./model2017-1_face12_nomouth.h5', 'r')
    # Select a specific weight from BFM
    triangles = np.asarray(bfm['shape/representer/cells'], dtype=np.int).T

    image = render(projection, image_colors, triangles, width, height)
    image, color = format_image(image, normalize=False)
    draw_pointcloud(image, color)

if __name__ == '__main__':
    texture_image(None, None, None, None)