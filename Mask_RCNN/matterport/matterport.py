
import os
import sys
import time
import numpy as np
import imgaug  # https://github.com/aleju/imgaug (pip3 install imageaug)

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class MatterPortConfig(Config):
    """Configuration for training on MatterPort.
    Derives from the base Config class and overrides values specific
    to the MatterPort dataset.
    """

    # Number of classes (including background)
    NUM_CLASSES = 10  # Matterport has 10 or more classes


############################################################
#  Dataset
############################################################

class MatterDataset(utils.Dataset):
    def load_matterport(self, dataset_dir, subset, class_ids=None,
                  class_map=None, return_coco=False, auto_download=False):

        if auto_download is True:
            self.auto_download(dataset_dir, subset)

        pass

    def load_mask(self, image_id):
        pass



############################################################
#  Matterport Evaluation
############################################################

def build_matterport_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    pass


def evaluate_matterport(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None):
    pass

############################################################
#  Training
############################################################


if __name__ == '__main__':
    pass