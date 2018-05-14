import os
import numpy as np
import h5py
from PIL import Image
import random
import sys
import time

import zipfile
import urllib.request
import shutil

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

class NYUDepthV2Dataset(utils.Dataset):
    def load_nyu_depth_v2(self, path):
        """Load the NYU dataset.
        """
        print("load dataset: %s" % (path))
        f = h5py.File(path)
        classes = {}
        i=1
        for c in f.get("names").value[0]:
            classes[i] = "".join([chr(j) for j in f[c].value])
            self.add_class("NYU", i, classes[i])
            i+=1

        dataset = []
        for i, (image, depth) in enumerate(zip(f['images'], f['depths'])):
            ra_image = image.transpose(2, 1, 0)
            ra_depth = depth.transpose(1, 0)
            re_depth = (ra_depth / np.max(ra_depth)) * 255.0
            image_labels = f['labels'][i,:,:].T
            #image_pil = Image.fromarray(np.uint8(ra_image))
            #depth_pil = Image.fromarray(np.uint8(re_depth))
            image_name = os.path.join("data", "nyu_datasets", "%05d.jpg" % (i))
            #image_pil.save(image_name)
            #depth_name = os.path.join("data", "nyu_datasets", "%05d.png" % (i))
            #depth_pil.save(depth_name)

            dataset.append((ra_image, re_depth, image_labels))

        random.shuffle(dataset)

        return dataset


    def load_mask(self, dataset, image_id):
        """Load instance masks for the given image.

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        rgb, depth, labels = dataset[image_id]
        image_instances_classes = np.unique(labels)

        instance_masks = []
        class_ids = []
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for c in image_instances_classes:

            m = (labels == c)
            # Some objects are so small that they're less than 1 pixel area
            # and end up rounded out. Skip those objects.
            if m.max() < 1:
                continue
            instance_masks.append(m)
            class_ids.append(c)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return rgb, mask, class_ids

if __name__ == '__main__':
    current_directory = os.getcwd()
    nyu_path = 'nyu_depth_v2_labeled.mat'
    dataset = NYUDepthV2Dataset()
    ds = dataset.load_nyu_depth_v2('nyu_depth_v2_labeled.mat')
    dataset.prepare()
    image_id = 2
    image, mask, class_ids = dataset.load_mask(ds, image_id)

    bbox = utils.extract_bboxes(mask)

    # Display image and additional stats
    print("image_id ", image_id, dataset.image_reference(image_id))
    log("image", image)
    log("mask", mask)
    log("class_ids", class_ids)
    log("bbox", bbox)
    # Display image and instances
    visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names)