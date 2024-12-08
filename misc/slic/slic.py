import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from skimage.segmentation import slic
from skimage import color
from skimage.util import img_as_float
import cv2
import PIL
from torch_geometric.data import Data

from utils import *

# ------------------- SLIC ------------------- #
IMAGE_DIR = './in/'
SIZE = (224, 224)
N_SEGMENTS = 750
COMPACTNESS = .5
RADIUS_FACTOR = 1/8
MAX_NEIGHBORS = 32

SCALE = 10
EDGE_WIDTH = 5

print('Loading data...')

# Load images from the directory
images = load_dir(IMAGE_DIR)  # Replace with the path to your image directory
images = [cv2.resize(image, SIZE) for image in images]

for sample_image, i in zip(images, range(len(images))):
    slicified = slicify(sample_image, n_segments=N_SEGMENTS, compactness=COMPACTNESS, radius=SIZE[0]*RADIUS_FACTOR, max_neighbors=MAX_NEIGHBORS)

    print(slicified)

    slicified_image = superpixels_to_2d_image(rec=slicified, scale=SCALE, edge_width=EDGE_WIDTH, size_tuple=SIZE, bg=sample_image)
    slicified_graph = superpixels_to_2d_image(rec=slicified, scale=SCALE, edge_width=EDGE_WIDTH, size_tuple=SIZE, bg=None)

    # plot the 3 images
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(sample_image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(slicified_image)
    plt.title('Superpixels')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(slicified_graph)
    plt.title('Graph Connectivity')
    plt.axis('off')

    plt.savefig(f'./slic_{i}.png')

    # plt.show()
