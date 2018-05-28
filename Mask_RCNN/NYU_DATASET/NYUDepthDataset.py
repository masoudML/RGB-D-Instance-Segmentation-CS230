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
import pandas as pd
import imgaug
import time
from collections import defaultdict
from mrcnn.config import Config
from coco.coco import CocoConfig
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

command = 'train'
COCO_MODEL_PATH = os.path.join(PROJ_DIR, "mask_rcnn_coco.h5")
COCO_NYU_CLASS_MAP_PATH = os.path.join(PROJ_DIR, "coco_nyu_classes_map.csv")
coco_nyu_class_map = pd.read_csv(COCO_NYU_CLASS_MAP_PATH)

# Directory to save logs and model checkpoints, if not provided
DEFAULT_LOGS_DIR = os.path.join(PROJ_DIR, "logs")
NYU_DATASET_DIR = os.path.join(PROJ_DIR, "NYU_DATASET")
NYU_DATASET_PATH = NYU_DATASET_DIR+'/nyu_depth_v2_labeled.mat'


class NUYDataObject():
    class __NUYDataObject:
        def __init__(self):
            self.dataset_size = [.5, .1, .01,]
            self.datasets = {}
            self.classes = dict(zip(coco_nyu_class_map.COCO_CLASS_ID, coco_nyu_class_map.CLASS_NAME))
            self.nyu_coco_map = dict(zip(coco_nyu_class_map.NYU_CLASS_ID, coco_nyu_class_map.COCO_CLASS_ID))
            self.images_masks_map = {}
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
            dev_dataset = dict(zip(range(0, dev_data_size),range(train_data_size, train_data_size + dev_data_size) ))
            test_data_size = int(self.dataset_size[2] * img_num)
            test_dataset = dict(zip(range(0, test_data_size), range(train_data_size + dev_data_size, train_data_size + dev_data_size + test_data_size)))

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
                self.images_masks_map[i] = image_dict
               # if (i <= self.dataset_size[0]):
               #     train_dataset.append(image_dict)
               # elif (i > self.dataset_size[0] and i <= self.dataset_size[0] + self.dataset_size[1]):
               #     dev_dataset.append(image_dict)
               # else:
               #     test_dataset.append(image_dict)

            ''' random.shuffle(train_dataset)
            random.shuffle(dev_dataset)
            random.shuffle(test_dataset)'''
            self.datasets['train'] = train_dataset
            self.datasets['dev'] = dev_dataset
            self.datasets['test'] = test_dataset
            f.close()


        def load_image(self, image_id, dstype='train', imagetype='rgb'):

            img_id = self.datasets[dstype][image_id]
            image = self.images_masks_map[img_id][imagetype]
            '''
            f = h5py.File(NYU_DATASET_PATH, 'r')
            if imagetype == 'rgb':
                image = f['images'][img_id, :, :, :].T  # image.transpose(2, 1, 0)
            elif imagetype == 'depth':
                ra_depth = f['depths'][img_id, :, :].T  # depth.transpose(1, 0)
                image = (ra_depth / np.max(ra_depth)) * 255.0
            else:
                image = f['labels'][img_id, :, :].T

            f.close()
            '''
            return image


        def getClasses(self):
            return self.classes

        def NYUtoCOCOClassId(self, nyu_cls_id):
            if nyu_cls_id == 0:
                return 0
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

    ## backbone

    STEPS_PER_EPOCH = 1000

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2
    LEARNING_RATE = 0.01
    IMAGE_MAX_DIM = 256
    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80

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
        #import pandas as pd
        #writer = pd.ExcelWriter('nyu_classes.xlsx')
        #df = pd.DataFrame.from_dict(list(classes.items()))
        #df.to_excel(writer,'sheet1')
       # writer.save()
        if (forEval):
            categories = []
            annotations = []
            images = []
            for index, row in coco_nyu_class_map.iterrows():
                category = {'supercategory': row['SUPERCATECORY'],
                            'id': row['COCO_CLASS_ID'],
                            'name': row['CLASS_NAME']
                }
                print(row)
                categories.append(category)

            i = 0
            for k, v in self.nyu_do.datasets[self.type].items():
                image_dict ={'id':k,
                             'image_id':v}
                images.append(image_dict)
                image = self.load_image(k)
                mask, class_ids = self.load_mask(k)
                bbox = utils.extract_bboxes(mask)
                j = 0
                for c in class_ids:
                    ann = {'id': i, # id
                                'image_id': k, # image id in the dataset
                                'bbox': bbox[j],
                                'category_id':c,
                                'iscrowd':0,
                                'area' : 10000} # area is just not to ignore the ground truth box
                    print(bbox[j])
                    j += 1
                    i += 1
                    annotations.append(ann)

                self.dataset_dict = {'annotations':annotations,
                                    'images': images,
                                    'categories': categories}
                self.add_image("COCO", image_id=k, path=NYU_DATASET_PATH)

            for k, v in classes.items():
                self.add_class("COCO", k, v)

            coco = NYU(self)
            return coco

        for k, v in classes.items():
            self.add_class("NYU", k, v)
        for k in self.nyu_do.datasets[self.type].keys():
            self.add_image("NYU", image_id=k,path=NYU_DATASET_PATH)


    def load_image(self, image_id):
        return self.nyu_do.load_image(image_id, self.type)


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

def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": class_id,
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results


def evaluate_model(model, dataset, nyu, eval_type="bbox", image_ids=None):
    image_ids = image_ids or dataset.image_ids

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        # Cast masks to uint8 because COCO tools errors out on bool
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"],
                                           r["masks"].astype(np.uint8))
        results.extend(image_results)
        print(i)

    # Load results. This modifies results with additional attributes.
    nyu_results = nyu.loadRes(results)

    # Evaluate
    cocoEval = COCOeval(nyu, nyu_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)


if __name__ == '__main__':

    current_directory = os.getcwd()
    config = NYUConfig()
    nyu_path = 'nyu_depth_v2_labeled.mat'
    nyu_ds_train = NYUDepthDataset(type='test')
    nyu_ds_train.load_nyu_depth_v2('nyu_depth_v2_labeled.mat')
    nyu_ds_train.prepare()
    if (command == "data_inspect"):
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

    if(command == 'train'):


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
                    augmentation=None)

        model.save_weights("model.h5")

        # Evaluate Model
    if (command == 'evaluate'):
        class InferenceConfig(CocoConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
        #config.BATCH_SIZE = 1
        #config.GPU_COUNT = 1
        #config.IMAGES_PER_GPU = 1
        #config.DETECTION_MIN_CONFIDENCE = 0
        config.display()
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=DEFAULT_LOGS_DIR)

        print("Loading weights ", COCO_MODEL_PATH)
        model.load_weights(COCO_MODEL_PATH, by_name=True)

        dataset_test = NYUDepthDataset(type='test')
        nyu = dataset_test.load_nyu_depth_v2('nyu_depth_v2_labeled.mat', forEval=True)
        dataset_test.prepare()
        evaluate_model(model, dataset_test,nyu)