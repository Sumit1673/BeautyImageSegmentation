import sys, os

import matplotlib.pyplot as plt 

# Root directory of the project
ROOT_DIR = os.path.abspath(".")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

from inferencing import *


if __name__ == "__main__":
    config = InferenceConfig()

    image_path = "assets/beauty_haircolor07_0.jpg"

    model = load_model(
        model_weights_path="logs/beauty20210526T0450/mask_rcnn_beauty_0030.h5")
    
    mask, bbox, label= get_inferencing(
        model, image_path=image_path, config=config)
    
    

    matplotlib.image.imsave('name.png', mask[:,:,0], cmap="gray")
