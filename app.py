import sys, os
import json, numpy as np
import matplotlib.pyplot as plt 
from matplotlib.image import imsave
from threading import Thread

# Root directory of the project
ROOT_DIR = os.path.abspath(".")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

from inferencing import *


def save_data(mask, bbox, label):

    if label == "nails":
        mask = sum(np.dsplit(mask, mask.shape[2]))


    # final_mask_img = 
    
    imsave('test_images/name.png', mask[:,:,0], cmap="gray")

    bbox_dict = {i: box.tolist() for i, box in enumerate(bbox)}
    data = {"label": label, "bbox": bbox_dict}

    with open('temp.json', 'w') as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    config = InferenceConfig()

    image_path = "samples/datasets/train/beauty_french-manicure_0_21.jpg"

    model = load_model(
        model_weights_path="logs/beauty20210526T0450/mask_rcnn_beauty_0030.h5")
    
    mask, bbox, label= get_inferencing(
        model, image_path=image_path, config=config)  

    
    t1 = Thread(target=save_data, args=[mask, bbox, label])
    t1.start() 

    # bbox_dict = {i: box for i, box in enumerate(bbox)}
    # data = {"label": label, "bbox": bbox_dict}

    # with open('temp.json', 'w') as fp:
    #     json.dump(data, fp)




    # imsave('test_images/name.png', mask[:,:,0], cmap="gray")
    