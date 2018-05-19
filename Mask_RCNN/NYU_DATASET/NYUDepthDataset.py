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
print ('\n'.join(sys.path))
# Import Mask RCNN
sys.path.append(PROJ_DIR)  # To find local version of the library
from mrcnn import visualize
from mrcnn.model import log
from mrcnn import model as modellib, utils
import pandas as pd
import imgaug

from mrcnn.config import Config
from coco.coco import CocoConfig
command = 'train'
COCO_MODEL_PATH = os.path.join(PROJ_DIR, "mask_rcnn_coco.h5")
COCO_NYU_CLASS_MAP_PATH = os.path.join(PROJ_DIR, "coco_nyu_classes_map.csv")

# Directory to save logs and model checkpoints, if not provided
DEFAULT_LOGS_DIR = os.path.join(PROJ_DIR, "logs")
NYU_DATASET_DIR = os.path.join(PROJ_DIR, "NYU_DATASET")
NYU_DATASET_PATH = NYU_DATASET_DIR+'/nyu_depth_v2_labeled.mat'
class NUYDataObject():
    class __NUYDataObject:
        def __init__(self):
            self.dataset_size = [.8, .1, .1]
            self.datasets = {}
            self.coco_nyu_class_map = pd.read_csv(COCO_NYU_CLASS_MAP_PATH)
            self.classes = dict(zip(self.coco_nyu_class_map.COCO_CLASS_ID, self.coco_nyu_class_map.CLASS_NAME))
            self.nyu_coco_map = dict(zip(self.coco_nyu_class_map.NYU_CLASS_ID, self.coco_nyu_class_map.COCO_CLASS_ID))

        def __str__(self):
            pass

        def load_dataset(self, path):
            print("load dataset: %s" % (NYU_DATASET_PATH))
            if ('train' in self.datasets.keys()):
                return

            f = h5py.File(NYU_DATASET_PATH, 'r')

            img_num = len(f['images'])
            train_data_size = int(self.dataset_size[0] * img_num)
            train_dataset = dict(zip(range(0, train_data_size), range(0, train_data_size)))
            dev_data_size = int(self.dataset_size[1] * img_num)
            dev_dataset = dict(zip(range(train_data_size, train_data_size + dev_data_size), range(0, dev_data_size)))
            test_data_size = int(self.dataset_size[2] * img_num)
            test_dataset = dict(zip(range(train_data_size + dev_data_size, img_num), range(0, test_data_size)))

            '''
            
            
            for i, (image, depth) in enumerate(zip(f['images'], f['depths'])):
                rgb_image = f['images'][i,:,:,:].T #image.transpose(2, 1, 0)
                ra_depth = f['depths'][i,:,:].T #depth.transpose(1, 0)
                depth_image = (ra_depth / np.max(ra_depth)) * 255.0
                image_labels = f['labels'][i, :, :].T
                # image_pil = Image.fromarray(np.uint8(ra_image))
                # depth_pil = Image.fromarray(np.uint8(re_depth))
                #image_name = os.path.join("data", "nyu_datasets", "%05d.jpg" % (i))
                # image_pil.save(image_name)
                # depth_name = os.path.join("data", "nyu_datasets", "%05d.png" % (i))
                # depth_pil.save(depth_name)
                image_dict = {"rgb": rgb_image, 'depth': depth_image, 'labels': image_labels}
                if (i <= self.dataset_size[0]):
                    train_dataset.append(image_dict)
                elif (i > self.dataset_size[0] and i <= self.dataset_size[0] + self.dataset_size[1]):
                    dev_dataset.append(image_dict)
                else:
                    test_dataset.append(image_dict)

            random.shuffle(train_dataset)
            random.shuffle(dev_dataset)
            random.shuffle(test_dataset)'''
            self.datasets['train'] = train_dataset
            self.datasets['dev'] = dev_dataset
            self.datasets['test'] = test_dataset
            f.close()


        def load_image(self, image_id, dstype='train', imagetype='rgb'):

            img_id = self.datasets[dstype][image_id]
            f = h5py.File(NYU_DATASET_PATH, 'r')
            if imagetype == 'rgb':
                image = f['images'][img_id, :, :, :].T  # image.transpose(2, 1, 0)
            elif imagetype == 'depth':
                ra_depth = f['depths'][img_id, :, :].T  # depth.transpose(1, 0)
                image = (ra_depth / np.max(ra_depth)) * 255.0
            else:
                image = f['labels'][img_id, :, :].T

            f.close()
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


class NYUConfig(CocoConfig):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "NYUDepth"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80

class NYUDepthDataset(utils.Dataset):
    def __init__(self, type='train'):
        super(NYUDepthDataset, self).__init__()
        self.type = type
        self.nyu_do = NUYDataObject()

    def load_nyu_depth_v2(self, path):
        """Load the NYU dataset.
        """
        self.nyu_do.load_dataset(path)
        classes = self.nyu_do.getClasses()
        #import pandas as pd
        #writer = pd.ExcelWriter('nyu_classes.xlsx')
        #df = pd.DataFrame.from_dict(list(classes.items()))
        #df.to_excel(writer,'sheet1')
       # writer.save()
        for k, v in classes.items():
            self.add_class("NYU", k, v)
        for k in self.nyu_do.datasets[self.type].keys():
            self.add_image("NYU", image_id=k,path=NYU_DATASET_PATH)


    def load_image(self, image_id):
        return self.nyu_do.load_image(image_id)


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

    def getClasses(self):
        return self.nyu_do.getClasses()
if __name__ == '__main__':

    current_directory = os.getcwd()
    nyu_path = 'nyu_depth_v2_labeled.mat'
    nyu_ds_train = NYUDepthDataset()
    nyu_ds_train.load_nyu_depth_v2('nyu_depth_v2_labeled.mat')
    nyu_ds_train.prepare()
    if (command == "data_inspect"):
        nyu_ds_train.prepare()
        image_id = 257
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

    if(command == 'train'):

        config = NYUConfig()
        nyu_ds_dev = NYUDepthDataset(type='dev')
        nyu_ds_dev.load_nyu_depth_v2('nyu_depth_v2_labeled.mat')
        nyu_ds_dev.prepare()

        augmentation = imgaug.augmenters.Fliplr(0.5)
        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        config.display()
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=DEFAULT_LOGS_DIR)

        print("Loading weights ", COCO_MODEL_PATH)
        model.load_weights(COCO_MODEL_PATH, by_name=True)

        print("Fine tune Resnet stage 4 and up")
        model.train(nyu_ds_train, nyu_ds_dev,
                    learning_rate=config.LEARNING_RATE,
                    epochs=10,
                    layers='4+',
                    augmentation=augmentation)