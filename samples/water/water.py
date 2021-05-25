"""
Train Mask R-CNN on the water segmentation dataset by Kaggle:
https://www.kaggle.com/gvclsu/water-segmentation-dataset
"""

import os
import sys
import time
import numpy as np
import imgaug

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask R-CNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


class WaterConfig(Config):
    NAME = "water-segmentation"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + water-segmentation


class WaterDataset(utils.Dataset):

    def load_water(self, dataset_dir, subset):
        """
        Load a subset of the Water Segmentation Dataset
        dataset_dir: Root directory of the dataset
        subset: Subset to load: train or val
        """
        self.add_class("water", 1, "water")

        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

    def load_mask(self, image_id):
        pass
