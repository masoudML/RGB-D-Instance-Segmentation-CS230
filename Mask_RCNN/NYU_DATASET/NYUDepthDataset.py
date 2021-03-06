import os
import numpy as np
import h5py
import random
import sys
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
PROJ_DIR = os.path.abspath("../")
print(PROJ_DIR)
sys.path.append(PROJ_DIR)
# Import Mask RCNN
sys.path.append(PROJ_DIR)  # To find local version of the library
from mrcnn import visualize
from mrcnn.model import log
from mrcnn import model as modellib, utils
from mrcnn import modeldepth as modellibDepth

import pandas as pd
import imgaug
import time
from skimage.color import gray2rgb
from collections import defaultdict
from mrcnn.config import Config
from coco.coco import CocoConfig
from pycocotools.coco import COCO


command = 'traindepth'
COCO_MODEL_PATH = os.path.join(PROJ_DIR, "mask_rcnn_coco.h5")
COCO_NYU_CLASS_MAP_PATH = os.path.join(PROJ_DIR, "coco_nyu_classes_map.csv")
coco_nyu_class_map = pd.read_csv(COCO_NYU_CLASS_MAP_PATH)

# Directory to save logs and model checkpoints, if not provided
DEFAULT_LOGS_DIR = os.path.join(PROJ_DIR, "logs")
NYU_DATASET_DIR = os.path.join(PROJ_DIR, "NYU_DATASET")
NYU_DATASET_PATH = NYU_DATASET_DIR+'/nyu_depth_v2_labeled.mat'
SAVED_MODELS_DIR = os.path.join(PROJ_DIR, "models")

# defaults
stages = 'all' # training stages/layers
epochs = 20  # number of training epochs
augmentation = None # no augmentation
learning_rate = 0.001

# AP Threshold
AP_THRESHOLD  = 0.0001
IOU_THRESHOLD = 0.5

class NUYDataObject():
    class __NUYDataObject:
        def __init__(self):
            self.dataset_size = [.8, .1, .1]
            self.datasets = {}
            self.classes = dict(zip(coco_nyu_class_map.COCO_CLASS_ID, coco_nyu_class_map.CLASS_NAME))
            self.nyu_coco_map = dict(zip(coco_nyu_class_map.NYU_CLASS_ID, coco_nyu_class_map.COCO_CLASS_ID))
            self.images_masks_map = {}
        def __str__(self):
            pass

        def load_dataset(self, path):
            if ('train' in self.datasets.keys()):
                return

            print("load dataset: %s" % (NYU_DATASET_PATH))
            f = h5py.File(NYU_DATASET_PATH, 'r')

            img_num = len(f['images'])
            img_nums = np.arange(img_num)
            random.shuffle(img_nums)
            train_data_size = int(self.dataset_size[0] * img_num)
            train_dataset = dict(zip(range(0, train_data_size), img_nums[0:train_data_size]))
            dev_data_size = int(self.dataset_size[1] * img_num)
            dev_dataset = dict(zip(range(0, dev_data_size),img_nums[train_data_size:(train_data_size + dev_data_size)] ))
            test_data_size = int(self.dataset_size[2] * img_num)
            test_dataset = dict(zip(range(0, test_data_size), img_nums[(train_data_size + dev_data_size): (train_data_size + dev_data_size + test_data_size)]))

            for i, (image, depth) in enumerate(zip(f['images'], f['depths'])):
                rgb_image = f['images'][i,:,:,:].T #image.transpose(2, 1, 0)
                ra_depth = f['depths'][i,:,:].T #depth.transpose(1, 0)
                depth_image = (ra_depth / np.max(ra_depth)) * 255.0
                image_labels = f['labels'][i, :, :].T
                image_dict = {"rgb": rgb_image, 'depth': depth_image, 'labels': image_labels}
                self.images_masks_map[i] = image_dict

            self.datasets['train'] = train_dataset
            self.datasets['dev'] = dev_dataset
            self.datasets['test'] = test_dataset
            f.close()


        def load_image(self, image_id, dstype='train', imagetype='rgb'):

            img_id = self.datasets[dstype][image_id]
            image = self.images_masks_map[img_id][imagetype]

            return image


        def getClasses(self):
            return self.classes

        def NYUtoCOCOClassId(self, nyu_cls_id):
            ret = -1
            if nyu_cls_id in self.nyu_coco_map.keys():
                ret = self.nyu_coco_map[nyu_cls_id]
            return ret

    instance = None

    def __new__(cls):  # __new__ always a classmethod
        if not NUYDataObject.instance:
            NUYDataObject.instance = NUYDataObject.__NUYDataObject()
        return NUYDataObject.instance

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def __setattr__(self, name):
        return setattr(self.instance, name)


class NYU(COCO):
    def __init__(self, dataset):
        super(NYU, self).__init__()

        tic = time.time()
        self.dataset = dataset.dataset_dict
        self.createIndex()
        print('Done (t={:0.2f}s)'.format(time.time() - tic))


class NYUDepthDataset(utils.Dataset):
    def __init__(self, type='train'):
        super(NYUDepthDataset, self).__init__()
        self.type = type
        self.nyu_do = NUYDataObject()

    def load_nyu_depth_v2(self, path, forEval=False):
        """Load the NYU dataset.
        """
        self.nyu_do.load_dataset(path)
        classes = self.nyu_do.getClasses()

        for k, v in classes.items():
            self.add_class("NYU", k, v)
        for k in self.nyu_do.datasets[self.type].keys():
            self.add_image("NYU", image_id=k,path=NYU_DATASET_PATH)


    def load_image(self, image_id):
        return self.nyu_do.load_image(image_id, self.type)

    def load_image_rgb_depth(self, image_id):
        return (self.nyu_do.load_image(image_id, self.type, imagetype='rgb'), \
                gray2rgb(self.nyu_do.load_image(image_id, self.type, imagetype='depth')))

    def load_mask(self, image_id, ds='train'):
        """Load instance masks for the given image.

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        labels = self.nyu_do.load_image(image_id, dstype=self.type, imagetype='labels')
        image_instances_classes = np.unique(labels)

        instance_masks = []
        class_ids = []
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for c in image_instances_classes:
            coco_cls_id = self.nyu_do.NYUtoCOCOClassId(c)
            if (coco_cls_id == -1):
                continue
            m = (labels == c)
            # Some objects are so small that they're less than 1 pixel area
            # and end up rounded out. Skip those objects.
            if m.max() < 1:
                continue
            instance_masks.append(m)
            class_ids.append(coco_cls_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            return np.array(instance_masks), np.array(class_ids, dtype=np.int32)

    def getClasses(self):
        return self.nyu_do.getClasses()

def get_ax(rows=1, cols=1, size=16):
    import matplotlib.pyplot as plt
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

def evaluate_model(type='RGB'):

    if(type=='RGB'):
        modlib = modellib
    else:
        modlib = modellibDepth

    model = modlib.MaskRCNN(mode="inference", config=config,
                              model_dir=DEFAULT_LOGS_DIR)

    if args.model:
        print("Loading weights ", DEFAULT_LOGS_DIR + '/' + args.model)
        model.load_weights(DEFAULT_LOGS_DIR + '/' + args.model, by_name=True)

    dataset_test = NYUDepthDataset(type='test')
    nyu = dataset_test.load_nyu_depth_v2('nyu_depth_v2_labeled.mat')
    dataset_test.prepare()
    mAP = 0
    APs = []
    #image_id = random.choice(dataset_test.image_ids)
    for image_id in dataset_test.image_ids:
        if(type=='RGB'):
            image, image_meta, gt_class_id, gt_bbox, gt_mask = \
                modlib.load_image_gt(dataset_test, config, image_id, use_mini_mask=False)
        else:
            image, depthimage, image_meta, gt_class_id, gt_bbox, gt_mask = \
                modlib.load_images_gt(dataset_test, config, image_id, use_mini_mask=False)

        info = dataset_test.image_info[image_id]
        print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                               dataset_test.image_reference(image_id)))
        # Run object detection
        if(type=='RGB'):
            results = model.detect([image])
        else:
            results = model.detectWdepth([image],[depthimage])

        r = results[0]

        try:
            AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                                         r['rois'], r['class_ids'], r['scores'], r['masks'], iou_threshold=IOU_THRESHOLD)
            if(AP > AP_THRESHOLD):
                APs.append(AP)
                print('Image %d AP = : %f' % (image_id, AP))
                dataset_test.nyu_do.load_image(image_id)
        except:
            print('GT classes in the image are not covered')

    mAP = np.mean(np.array(APs))
    print('mAP : %f' % mAP)

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on NYU Dataset.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train','traindepth' , 'transfertrain', 'data_inspect' or 'evaluate' on NYU Dataset")
    parser.add_argument('--model', required=False,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file")
    parser.add_argument('--imgsize', required=False,
                        metavar="imagesize",
                        help="Resize image to size")
    parser.add_argument('--backbone', required=False,
                        metavar="backbone",
                        help="Backbone (resnet101, resnet50) ")
    parser.add_argument('--LR', required=False,
                        metavar="Learning Rate",
                        help="Learning Rate")
    parser.add_argument('--augment', required=False,
                        metavar="augmentation",
                        help="augmentation (True/False)")
    parser.add_argument('--early', required=False,
                        metavar="earlystopping",
                        help="Early Stopping (True/False)")
    parser.add_argument('--epochs', required=False,
                        metavar="epochs",
                        help="Number of Epochs")
    parser.add_argument('--trainstages', required=False,
                        metavar="TrainStages",
                        help="stages to train")

    parser.add_argument('--lmbda', required=False,
                        metavar="Regularization Parameter (Lmdba)",
                        help="Regularization Parameter (Lmdba)")


    parser.add_argument('--eval', required=False,
                        metavar="Evaluate",
                        help="Evaluate Model")

    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("backbone: ", args.backbone)
    print("image config size: ", args.imgsize)


    class NYUConfig(CocoConfig):
        """Configuration for training on NYU Dataset
        Derives from the base COCO Config class and overrides values specific
        to the NYU dataset.
        """
        # Give the configuration a recognizable name
        NAME = "NYUDepth"

        if (args.backbone):
            BACKBONE = args.backbone

        if (args.imgsize):
            img_size = int(args.imgsize)
            IMAGE_MAX_DIM = IMAGE_MIN_DIM = img_size
            if (img_size < 512):
                RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

        if (args.command == 'evaluate' or args.command == 'evaluatedepth'):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        if(args.lmbda):
            WEIGHT_DECAY = float(args.lmbda)
        if (args.LR):
            LEARNING_RATE = float(args.LR)

        STEPS_PER_EPOCH = 100
        VALIDATION_STEPS = 10
        TRAIN_ROIS_PER_IMAGE = 100


    config = NYUConfig()
    config.display()

    if (args.augment):
        augmentation = imgaug.augmenters.Fliplr(0.5)
    if(args.trainstages):
        stages = args.trainstages
    if(args.epochs):
        epochs = int(epochs)

    nyu_path = 'nyu_depth_v2_labeled.mat'
    nyu_ds_train = NYUDepthDataset(type='train')
    nyu_ds_train.load_nyu_depth_v2('nyu_depth_v2_labeled.mat')
    nyu_ds_train.prepare()
    nyu_ds_dev = NYUDepthDataset(type='dev')
    nyu_ds_dev.load_nyu_depth_v2('nyu_depth_v2_labeled.mat')
    nyu_ds_dev.prepare()


    if args.command == "data_inspect":
        nyu_ds_train.prepare()
        image_id = 0
        image = nyu_ds_train.load_image(image_id)
        mask, class_ids = nyu_ds_train.load_mask(image_id)

        bbox = utils.extract_bboxes(mask)

        # Display image and additional stats
        print("image_id ", image_id, nyu_ds_train.image_reference(image_id))
        log("RGB image", image)
        log("Depth image", image)
        log("mask", mask)
        log("class_ids", class_ids)
        log("bbox", bbox)
        # Display image and instances
        visualize.display_instances(image, bbox, mask, class_ids, nyu_ds_train.class_names)

    if args.command == 'traintransfer':
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=DEFAULT_LOGS_DIR)

        print("Loading weights ", COCO_MODEL_PATH)
        model.load_weights(COCO_MODEL_PATH, by_name=True)
        model.train(nyu_ds_train, nyu_ds_dev,
                    learning_rate=config.LEARNING_RATE,
                    epochs=epochs,
                    layers=stages,
                    augmentation=augmentation)


    if args.command == 'train':
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=DEFAULT_LOGS_DIR)

        if args.model:
            print("Loading weights ", DEFAULT_LOGS_DIR + '/' + args.model)
            model.load_weights(DEFAULT_LOGS_DIR+'/'+args.model, by_name=True)

        model.train(nyu_ds_train, nyu_ds_dev,
                    learning_rate=config.LEARNING_RATE,
                    epochs=epochs,
                    layers=stages,
                    augmentation=augmentation)


    if args.command == "traindepth":
        model = modellibDepth.MaskRCNN(mode="training", config=config,
                                  model_dir=DEFAULT_LOGS_DIR)

        if args.model:
            print("Loading weights ", DEFAULT_LOGS_DIR + '/' + args.model)
            model.load_weights(DEFAULT_LOGS_DIR+'/'+args.model, by_name=True)

        model.train(nyu_ds_train, nyu_ds_dev,
                    learning_rate=config.LEARNING_RATE,
                    epochs=epochs,
                    layers=stages,
                    augmentation=augmentation)


    # Evaluate Model - h5
    if args.command == "evaluate":
        evaluate_model()

    if args.command == "evaluatedepth":
        evaluate_model(type='depth')