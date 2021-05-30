import os
import sys
import itertools
import math
import logging
import json
import re
import random
from collections import OrderedDict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

from samples.beauty import beauty

if __name__ == "__main__":

    config = beauty.BeautyConfig()
    BEAUTY_DIR = os.path.join(ROOT_DIR, "datasets")

    # Load dataset
    # Get the dataset from the releases page
    # https://github.com/matterport/Mask_RCNN/releases
    dataset = beauty.BeautyDataset()
    dataset.load_beauty(BEAUTY_DIR, "train")

    # Must call before using the dataset
    dataset.prepare()

    print("Image Count: {}".format(len(dataset.image_ids)))
    print("Class Count: {}".format(dataset.num_classes))
    for i, info in enumerate(dataset.class_info):
        print("{:3}. {:50}".format(i, info['name']))

    image_ids = np.random.choice(dataset.image_ids, 4)
    # print(image_ids)
    for image_id in image_ids:
        image = dataset.load_image(image_id)
        mask, class_ids = dataset.load_mask(image_id)
        print(class_ids)
        visualize.display_top_masks(image, mask, class_ids, dataset.class_names)

plt.show()

