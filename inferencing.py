import os
import sys
import scipy
import tensorflow as tf
import skimage.color
import skimage.io
import skimage.transform
from collections import OrderedDict
from mrcnn import model
from mrcnn import utils
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon

# # Root directory of the project
# ROOT_DIR = os.path.abspath(".")

# # Import Mask RCNN
# sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

from samples.beauty import beauty
config = beauty.BeautyConfig()


class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1




def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


def load_model(*, model_weights_path, device_type="/cpu:0"):
    with tf.device(device_type):
        model = modellib.MaskRCNN(mode="inference", model_dir=None,
                                config=config)

        model.load_weights(model_weights_path, by_name=True)
        return model
    # except (TypeError, FileNotFoundError):
    #     print("[ERROR] Invalid input")

def load_image_for_mask_rcnn(image_path):
    """Load the specified image and return a [H,W,3] Numpy array.
    """

    image = skimage.io.imread(image_path)
    # If grayscale. Convert to RGB for consistency.
    if image.ndim != 3:
        image = skimage.color.gray2rgb(image)
    # If has an alpha channel, remove it for consistency
    if image.shape[-1] == 4:
        image = image[..., :3]
        
    return image

def resize_image(image, config):
    image, window, scale, padding, crop = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE
    )

    return image
def get_prediction(model, image):
    return model.detect([image], verbose=0)

def get_inferencing(model, *, image_path, config):
    """Takes the MASK-RCNN model and return the mask and bounding of an image

    Args:
        model ([MASKRCNN]): mask rcnn model object
        image_path ([str]): complete path of the image

    Returns:
        [np array]: Mask of the image
        [np array]: Bounding box coordinates
        [str]: Class Name
    """
    LABELS = ['BG', 'hair', 'nails']

    image = load_image_for_mask_rcnn(image_path)

    resized_image =  resize_image(image, config)

    results = get_prediction(model, resized_image)
    # exit()

    # Display results
    ax = get_ax(1)
    r = results[0]
   
    return r['masks'], r['rois'], LABELS[r['class_ids'][0]]


if __name__ == "__main__":
    config = InferenceConfig()
