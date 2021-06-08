# Improting Image class from PIL module
from PIL import Image
# Enable Dominant Color
from sklearn.cluster import KMeans
import cv2
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
import numpy as np
import matplotlib.pyplot as plt


def find_dominant_colors(image_path, bbox):
    return crop_image(image_path, bbox)

def crop_image(image_path, bbox):

    colors = []       

    im = Image.open(image_path)
    for i_bbox in range(bbox.shape[0]): 
    # Setting the points for cropped image
        box = bbox[i_bbox]
        left = box[1]
        top = box[0]
        right = box[-1]
        bottom = box[-2]
        im_crop = im.crop((left, top, right, bottom))
        np_im = np.array(im_crop)

        colors.append(get_colors(np_im, 3, show_chart=False))
    return colors


def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))



def get_colors(image, number_of_colors, show_chart):
    modified_image = image.reshape(image.shape[0]*image.shape[1], 3)
    
#     print(modified_image)
    clf = KMeans(n_clusters = number_of_colors)
    labels = clf.fit_predict(modified_image)
    
    counts = Counter(labels)
    # sort to ensure correct color percentage
    counts = dict(sorted(counts.items()))
    center_colors = clf.cluster_centers_
    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]
    total_pixels = sum(counts.values())
    percentage_list = [round(values/total_pixels, 3) for values in counts.values()]
    count_percentage = {key: percentage_list[hex_colors.index(key)] for key in hex_colors}
    count_percentage = dict(sorted(count_percentage.items(), key=lambda item: item[1], reverse = True))
    
    # print('percentage color', count_percentage)
    if show_chart:
        plt.figure(figsize = (8, 6))
        plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)
    return count_percentage  