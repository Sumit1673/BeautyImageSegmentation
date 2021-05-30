import os
import sys
import itertools
import math
import logging
import json
import re
import random
from collections import OrderedDict
from mrcnn import model
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

from samples.beauty import beauty
config = beauty.BeautyConfig()


def load_model(model_weights_path):
    try:
        if os.path.exists(model_weights_path):

            pass
    
    except Exception as e:
        print(e)
        