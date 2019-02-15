"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""

import os
import os.path
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import random
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

VID_CLASSES = (  # always index 0
    'n02691156', #1 airplane
    'n02419796', #2 antelope
    'n02131653', #3 bear
    'n02834778', #4 bicycle
    'n01503061', #5 bird
    'n02924116', #6 bus
    'n02958343', #7 car
    'n02402425', #8 cattle
    'n02084071', #9 dog
    'n02121808', #10 domestic_cat
    'n02503517', #11 elephant
    'n02118333', #12 fox
    'n02510455', #13 giant_panda
    'n02342885', #14 hamster
    'n02374451', #15 horse
    'n02129165', #16 lion
    'n01674464', #17 lizard
    'n02484322', #18 monkey
    'n03790512', #19 motorcycle
    'n02324045', #20 rabbit
    'n02509815', #21 red_panda
    'n02411705', #22 sheep
    'n01726692', #23 snake
    'n02355227', #24 squirrel
    'n02129604', #25 tiger
    'n04468005', #26 train
    'n01662784', #27 turtle
    'n04530566', #28 watercraft
    'n02062744', #29 whale
    'n02391049', #30 zebra
)
VID_CLASSES_name =(  # always index 0
    'airplane', #1 airplane
    'antelope', #2 antelope
    'bear', #3 bear
    'bicycle', #4 bicycle
    'bird', #5 bird
    'bus', #6 bus
    'car', #7 car
    'cattle', #8 cattle
    'dog', #9 dog
    'domestic_cat', #10 domestic_cat
    'elephant', #11 elephant
    'fox', #12 fox
    'giant_panda', #13 giant_panda
    'hamster', #14 hamster
    'horse', #15 horse
    'lion', #16 lion
    'lizard', #17 lizard
    'monkey', #18 monkey
    'motorcycle', #19 motorcycle
    'rabbit', #20 rabbit
    'red_panda', #21 red_panda
    'sheep', #22 sheep
    'snake', #23 snake
    'squirrel', #24 squirrel
    'tiger', #25 tiger
    'train', #26 train
    'turtle', #27 turtle
    'watercraft', #28 watercraft
    'whale', #29 whale
    'zebra', #30 zebra
)

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))


class AnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False, dataset_name='VOC0712'):

        self.dataset_name = dataset_name
        if (self.dataset_name == 'VOC0712'):
            self.class_to_ind = class_to_ind or dict(
               zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        elif (self.dataset_name == 'VID2017' or 'seqVID2017'):
            self.class_to_ind = class_to_ind or dict(
                zip(VID_CLASSES, range(len(VID_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height, img_id=None):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            if self.dataset_name=='VOC0712':
                difficult = int(obj.find('difficult').text) == 1
                if not self.keep_difficult and difficult:
                    continue
            name = obj.find('name').text.lower().strip()
            if name not in VID_CLASSES:
                continue
            bbox = obj.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            if(bndbox != []):
                res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root, image_sets, transform=None, target_transform=None,
                 dataset_name='VOC0712', set_file_name='train', seq_len=8, skip=False):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self.ids = list()
        self.video_size = list()
        self.seq_len = seq_len
        self.skip = skip
        if skip:
            print('Random collect data with a random skip')
        else:
            print('Random collect data continuously')
        if self.name =='VOC0712':
            self._annopath = os.path.join('%s', 'Annotations', '%s.xml')
            self._imgpath = os.path.join('%s', 'JPEGImages', '%s.jpg')
            for (year, name) in image_sets:
                rootpath = os.path.join(self.root, 'VOC' + year)
                for line in open(os.path.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                    self.ids.append((rootpath, line.strip()))
        elif self.name in ['VID2017', 'seqVID2017', 'VIDDET']:
            self._annopath = os.path.join('%s', 'Annotations', 'VID', image_sets, '%s.xml')
            self._imgpath = os.path.join('%s', 'Data', 'VID', image_sets, '%s.JPEG')
            rootpath = self.root[:-1]
            for line in open(os.path.join(rootpath, 'ImageSets', 'VID', set_file_name + '.txt')):
                pos = line.split(' ')
                self.ids.append((rootpath, pos[0][:-1])) if len(pos)==1 else self.ids.append((rootpath, pos[0]))
                if self.name == 'seqVID2017':
                    self.video_size.append(int(pos[1][:-1]))


    def __getitem__(self, index):

        if self.name == 'seqVID2017':
            im_list, gt_list, maskroi_list = self.pull_seqitem(index)
            return im_list, gt_list, maskroi_list
        else:
            loop_none_gt = True
            while loop_none_gt:
                im, gt, h, w, mask = self.pull_item(index)
                if len(gt) > 0:
                    loop_none_gt = False
                else:
                    index = index+1
            return im, gt, mask

    def __len__(self):
        return len(self.ids)

    def select_clip(self, video_id, video_size):
        target_list = list()
        img_list = list()

        if video_size <= self.seq_len:
            start_frame = 0
            repeat = self.seq_len // video_size
            residue = self.seq_len % video_size
            for i in range(start_frame, video_size):
                img_name = video_id[1]+'/'+str(i).zfill(6)
                for _ in range(repeat):
                    target_list.append(ET.parse(self._annopath % (video_id[0], img_name)).getroot())
                    img_list.append(cv2.imread(self._imgpath % (video_id[0], img_name)))
                if residue:
                    target_list.append(ET.parse(self._annopath % (video_id[0], img_name)).getroot())
                    img_list.append(cv2.imread(self._imgpath % (video_id[0], img_name)))
                    residue -= 1
        else:
            ## D Skip
            # skip = int(video_size / self.seq_len)
            # uniform_list = list(range(0, video_size, skip))
            # cast_list = random.sample(range(len(uniform_list)), len(uniform_list) - self.seq_len)
            # select_list = [x for x in uniform_list[::random.sample([-1, 1], 1)[0]] if
            #                uniform_list.index(x) not in cast_list]
            if self.skip:
                ## R Skip
                skip = random.randint(1, int(video_size / self.seq_len))
                start = random.randint(0, video_size - self.seq_len * skip)
                select_list = list(range(start, video_size, skip))[:self.seq_len]
            else:
                ## R Cont
                start = np.random.randint(video_size - self.seq_len)
                select_list = [x for x in range(start, start + self.seq_len)]

            img_name = [video_id[1]+'/'+str(i).zfill(6) for i in select_list]
            target_list, img_list = [ET.parse(self._annopath % (video_id[0], img_name)).getroot() for img_name in img_name], \
                                    [cv2.imread(self._imgpath % (video_id[0], img_name)) for img_name in img_name]


        return target_list, img_list

    def pull_seqitem(self, index):
        video_id = self.ids[index]
        video_size = self.video_size[index]

        target_list, img_list = self.select_clip(video_id, video_size)
        maskroi_list = list()

        # transform annotation
        for i, (target, img) in enumerate(zip(target_list, img_list)):
            height, width, channels = img.shape
            target_list[i] = self.target_transform(target, width, height)

        # for seq_tar, img in zip(target_list, img_list):
        #     for tar in seq_tar:
        #         x_min, y_min, x_max, y_max, cls = tar
        #         print(cls)
        #         img = cv2.rectangle(img, (int(x_min*width),int(y_min*height)), (int(x_max*width),int(y_max*height)), (255,0,0),10)
        #         cv2.imshow('test',img)
        #     cv2.waitKey(0)
            # if len(target_list[i]) == 0:
            #     target_list[i] = target_list[i-1]
            #     img_list[i] = img_list[i-1]

        # mirror = bool(np.random.randint(2))
        # expand = np.random.randint(2)
        # ratio = np.random.uniform(1, 4)
        for i, (target, img) in enumerate(zip(target_list, img_list)):
            target = np.array(target)
            # img, boxes, labels = self.transform(img, target[:, :4], target[:, 4],mirror=mirror, expand=expand*ratio)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])

            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
            img_list[i] = img
            target_list[i] = target

            maskroi = np.zeros([img.shape[0], img.shape[1]])
            for box in list(boxes):
                box[0] *= img.shape[1]
                box[1] *= img.shape[0]
                box[2] *= img.shape[1]
                box[3] *= img.shape[0]
                pts = np.array([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]], np.int32)
                maskroi = cv2.fillPoly(maskroi, [pts], 1)
            maskroi_list.append(np.expand_dims(maskroi, axis=0))

        return torch.from_numpy(np.array(img_list)).permute(0, 3, 1, 2), target_list, \
               torch.from_numpy(np.array(maskroi_list)).type(torch.FloatTensor)


    def pull_item(self, index):
        img_id = self.ids[index]

        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape
        maskroi = np.zeros([img.shape[0], img.shape[1]])

        if self.target_transform is not None:
            target = self.target_transform(target, width, height, img_id)
            if len(target) == 0:
                # target = np.array(target)
                img,_,_ = self.transform(img)
                img = img[:, :, (2, 1, 0)]
                return torch.from_numpy(img).permute(2, 0, 1), target, height, width, maskroi
            # box = target[0]
            # x_min, y_min, x_max, y_max, _ = box
            # print(x_min, y_min, x_max, y_max)
            # img = cv2.rectangle(img, (int(x_min*width),int(y_min*height)), (int(x_max*width),int(y_max*height)), (255,0,0),10)
            # cv2.imshow('test', cv2.resize(cv2.resize(img,(300,300)), (700,700)))
            # cv2.waitKey(1)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
            maskroi = np.zeros([img.shape[0], img.shape[1]])
            for box in list(boxes):
                box[0] *= img.shape[1]
                box[1] *= img.shape[0]
                box[2] *= img.shape[1]
                box[3] *= img.shape[0]
                pts = np.array([[box[0],box[1]],[box[2],box[1]],[box[2],box[3]],[box[0],box[3]]], np.int32)
                maskroi = cv2.fillPoly(maskroi, [pts], 1)
                # cv2.imshow('mask',cv2.resize(maskori, (700,700)))
            # cv2.waitKey(0)

        return torch.from_numpy(img).permute(2, 0, 1), target, height, width, torch.from_numpy(maskroi).type(torch.FloatTensor).unsqueeze(0)
        # return torch.from_numpy(img), target, height, width

    def pull_img_id(self, index):
        return self.ids[index]

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    masks = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
        masks.append(sample[2])
    return torch.stack(imgs, 0), targets, torch.stack(masks, 0)

def seq_detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    masks = []
    for sample in batch:
        imgs.append(sample[0])
        masks.append(sample[2])
        target = []
        for anno in sample[1]:
            target.append(torch.FloatTensor(anno))
        targets.append(target)

    return torch.stack(imgs, 0), targets, torch.stack(masks, 0)
