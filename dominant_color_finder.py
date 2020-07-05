from PIL import Image
from sklearn.cluster import KMeans
import cv2

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import numpy as np

from random import randint
from scipy import stats
from progress.bar import Bar
import argparse


def get_dominant_colors(image_name, confidence, desired_height):
    im = Image.open(image_name)
    im_copy = Image.open(image_name)
    (width, height) = im.size
    resize_factor = desired_height / height
    im = im.resize((int(width * resize_factor), int(height * resize_factor)))
    im_copy = im_copy.resize((int(width * resize_factor), int(height * resize_factor)))
    (width, height) = im.size

    pixel_list = []

    if confidence == 100:
        bar = Bar('Processing', max=height * width)
        for row in range(height):
            for col in range(width):
                (r, g, b) = im.getpixel((col, row))
                if r < 250 or g < 250 or b < 250:
                    pixel_list.append(im.getpixel((col, row)))
                    im_copy.putpixel((col, row), (127, 255, 0))
                else:
                    im_copy.putpixel((col, row), (255, 0, 0))
                bar.next()
        im_copy.save('checked.png')
        bar.finish()
    else:
        iterations = confidence / 100 * width * height
        iteration = 0
        checked_coordinates = []
        bar = Bar('Processing', max=iterations)
        max_iterations = 50
        while iteration < iterations:
            inner_iterations = 0
            while inner_iterations < max_iterations:
                random_coordinates = (randint(1, width - 1), randint(1, height - 1))
                if random_coordinates not in checked_coordinates:
                    (r, g, b) = im.getpixel(random_coordinates)
                    if r < 250 and g < 250 and b < 250:
                        break
                    else:
                        checked_coordinates.append(random_coordinates)
                        im_copy.putpixel(random_coordinates, (255, 0, 0))
                inner_iterations = inner_iterations + 1
            pixel_list.append(im.getpixel(random_coordinates))
            checked_coordinates.append(random_coordinates)
            im_copy.putpixel(random_coordinates, (127, 255, 0))
            iteration = iteration + 1
            bar.next()
        bar.finish()
        im_copy.save('checked.png')

    colors = np.array(pixel_list)
    return colors


def show_colors_plot(colors):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(colors[:, 0], colors[:, 1], colors[:, 2], c=colors / 255)
    ax.set_xlim((0, 255))
    ax.set_ylim((0, 255))
    ax.set_zlim((0, 255))
    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')

    plt.show()


def show_color_histogram(colors):
    cluster_count = 4
    kmeans = KMeans(n_clusters=cluster_count).fit(colors)
    CLUSTER_COLORS = kmeans.cluster_centers_

    # labels form 0 to no. of clusters
    numLabels = np.arange(0, cluster_count + 1)

    # create frequency count tables
    (hist, _) = np.histogram(kmeans.labels_, bins=numLabels)
    hist = hist.astype("float")
    hist /= hist.sum()

    # appending frequencies to cluster centers
    colors = CLUSTER_COLORS

    # descending order sorting as per frequency count
    colors = colors[(-hist).argsort()]
    hist = hist[(-hist).argsort()]

    # creating empty chart
    chart = np.zeros((50, 500, 3), np.uint8)
    start = 0

    # creating color rectangles
    for i in range(cluster_count):
        end = start + hist[i] * 500

        # getting rgb values
        r = colors[i][0]
        g = colors[i][1]
        b = colors[i][2]

        # using cv2.rectangle to plot colors
        cv2.rectangle(chart, (int(start), 0), (int(end), 50), (r, g, b), -1)
        start = end

    # display chart
    plt.figure()
    plt.axis("off")
    plt.imshow(chart)
    plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to image")
    ap.add_argument("-c", "--confidence", type=int, default=100, help="percent of pixels to be checked")
    ap.add_argument("-d", "--desiredheight", type=int, default=50, help="height to resize the image to")
    args = vars(ap.parse_args())

    colors = get_dominant_colors(image_name=args['image'], confidence=args['confidence'], desired_height=args['desiredheight'])
    # show_colors_plot(colors)
    show_color_histogram(colors)
