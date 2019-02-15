import numpy as np
import torch
import torch.utils.data as data
import os
import os.path
import sys
import cv2
import random


MOT_CLASSES = ('Pedestrian',)

class MOTDetection(data.Dataset):

    def __init__(self, root, image_sets, transform, dataset_name='MOT17Det', seq_len=8, skip=False):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.ids = list()
        self.video_size = list()
        self.seq_len = seq_len
        self.name = dataset_name
        self.skip = skip
        if skip:
            print('Random collect data with a random skip')
        else:
            print('Random collect data continuously')
        self._imgpath = os.path.join(self.root, '%s', 'img1', '%s.jpg')
        self._annotation = os.path.join(self.root, '%s', 'gt', 'gt_%s.txt')

        if self.name == 'MOT17Det':
            for line in open(os.path.join(self.root, 'ImageSet', image_sets+'.txt')):
                self.ids.append(tuple(line[:-1].split(' ')))
        elif self.name=='seqMOT17Det':
            for line in open(os.path.join(self.root, 'ImageSet', image_sets+'.txt')):
                split = line[:-1].split(' ')
                self.ids.append(split[0])
                self.video_size.append(int(split[1]))
        elif self.name == 'MOT15':
            for line in open(os.path.join(self.root, 'ImageSet', image_sets+'.txt')):
                self.ids.append(tuple(line[:-1].split(' ')))
        elif self.name=='seqMOT15':
            for line in open(os.path.join(self.root, 'ImageSet', image_sets+'.txt')):
                split = line[:-1].split(' ')
                self.ids.append(split[0])
                self.video_size.append(int(split[1]))

    def __getitem__(self, index):
        if self.name in ['seqMOT17Det', 'seqMOT15']:
            im_list, gt_list, maskroi_list = self.pull_seqitem(index)
            return im_list, gt_list, maskroi_list
        else:
            im, gt, h, w, mask = self.pull_item(index)
            return im, gt, mask

    def __len__(self):
        return len(self.ids)

    def select_clip(self, video_id, video_size):

        if self.skip:
            ## R Skip
            skip = random.randint(1, int(video_size / self.seq_len))
            start = random.randint(0, video_size - self.seq_len * skip)
            select_list = list(range(start+1, video_size+1, skip))[:self.seq_len]
        else:
            ## R Cont
            start = np.random.randint(video_size - self.seq_len)
            select_list = [x for x in range(start+1, start+1 + self.seq_len)]


        img_name = [str(i).zfill(6) for i in select_list]
        img_list = [cv2.imread(self._imgpath % (video_id, img)) for img in img_name]
        gt_list = [self._annotation % (video_id, img)  for img in img_name]
        target_list = [[] for _ in range(len(gt_list))]

        height, width, _ = img_list[0].shape
        for idx, gt_file in enumerate(gt_list):
            for line in open(gt_file):
                frame_num, id, x, y, w, h = line.split(',')[:6]
                x, y, w, h = x.split('.')[0], y.split('.')[0], w.split('.')[0], h.split('.')[0]
                target_list[idx].append([(np.int(x) - 1) / width, (np.int(y) - 1) / height, (np.int(x) + np.int(w) - 1) / width,
                               (np.int(y) + np.int(h) - 1) / height,
                               0])  # [x_min, y_min, x_max, y_max, class(0 for object here)]
        target_list = [np.array(o) for o in target_list]
        return target_list, img_list


    def pull_seqitem(self, index):
        video_id = self.ids[index]
        video_size = self.video_size[index]
        target_list, img_list = self.select_clip(video_id, video_size)
        # img_test=cv2.resize(img_list[0].copy(), (300,300))
        # cv2.imshow('img', cv2.resize(img_list[0], (700,700)))
        maskroi_list = list()
        for i, (target, img) in enumerate(zip(target_list, img_list)):
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
                # if i==0:
                #     img_test = cv2.rectangle(img_test, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0),1)
            # cv2.imshow('img', cv2.resize(img_test, (700,700)))
            # cv2.imshow('mask',cv2.resize(maskroi, (700,700)))
            # cv2.waitKey(0)
            maskroi_list.append(np.expand_dims(maskroi, axis=0))

        return torch.from_numpy(np.array(img_list)).permute(0, 3, 1, 2), target_list, \
           torch.from_numpy(np.array(maskroi_list)).type(torch.FloatTensor)

    def pull_item(self, index):

        img_id = self.ids[index]
        img = cv2.imread(self._imgpath % img_id)
        height, width, _ = img.shape

        target = []
        gt_file = self._annotation % img_id
        for line in open(gt_file):
            frame_num, id, x, y, w, h = line.split(',')[:6]
            x,y,w,h = x.split('.')[0],y.split('.')[0],w.split('.')[0],h.split('.')[0]
            target.append([(np.int(x)-1)/width, (np.int(y)-1)/height, (np.int(x)+np.int(w)-1)/width, (np.int(y)+np.int(h)-1)/height, 0])  # [x_min, y_min, x_max, y_max, class(0 for object here)]

        # for tar in target:
        #     x_min, y_min, x_max, y_max, _ = tar
        #     x_min = x_min * width + 1
        #     y_min = y_min * height + 1
        #     x_max = x_max * width + 1
        #     y_max = y_max * height + 1
        #     img = cv2.rectangle(img, (np.int(x_min),np.int(y_min)), (np.int(x_max),np.int(y_max)), (255,0,0),2)
        # cv2.imshow('test', cv2.resize(img, (700,700)))
        # cv2.waitKey(0)

        target = np.array(target)
        # print(img_id, target.shape)
        img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
        target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        # to rgb
        img = img[:, :, (2, 1, 0)]

        maskroi = np.zeros([img.shape[0], img.shape[1]])
        for box in list(boxes):
            box[0] *= img.shape[1]
            box[1] *= img.shape[0]
            box[2] *= img.shape[1]
            box[3] *= img.shape[0]
            pts = np.array([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]], np.int32)
            maskroi = cv2.fillPoly(maskroi, [pts], 1)
        # cv2.imshow('mask',cv2.resize(maskroi, (700,700)))
        # cv2.waitKey(0)

        return torch.from_numpy(img).permute(2, 0, 1), target, height, width, torch.from_numpy(maskroi).type(torch.FloatTensor).unsqueeze(0)
