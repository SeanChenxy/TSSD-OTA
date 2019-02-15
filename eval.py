"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data import VOCroot, VIDroot
# from data import VOC_CLASSES as labelmap
import torch.utils.data as data

from data import AnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES, VID_CLASSES, VID_CLASSES_name
from ssd import build_ssd

import sys
import os
import time
import argparse
import numpy as np
import pickle
import cv2

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--model_name', default='ssd300',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='../eval', type=str,
                    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--nms_threshold', default=0.45, type=float,
                    help=' nms threshold')
parser.add_argument('--top_k', default=200, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--dataset_name', default='VID2017', help='Which dataset')
parser.add_argument('--ssd_dim', default=300, type=int, help='ssd_dim 300 or 512')
parser.add_argument('--set_file_name', default='test', type=str,help='File path to save results')
parser.add_argument('--literation', default='20000', type=str,help='File path to save results')
parser.add_argument('--model_dir', default='../weights', type=str,help='Path to save model')
parser.add_argument('--detection', default='no', type=str2bool, help='detection or not')
parser.add_argument('--tssd',  default='lstm', type=str, help='ssd or tssd')
parser.add_argument('--gpu_id', default='2,3', type=str,help='gpu id')
parser.add_argument('--attention', default=False, type=str2bool, help='attention')
parser.add_argument('--tub', default=0, type=int, help='tubelet max size')
parser.add_argument('--tub_thresh', default=0.95, type=float, help='> : generate tubelet')
parser.add_argument('--tub_generate_score', default=0.7, type=float, help='> : generate tubelet')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

set_type = args.set_file_name.split('_')[0]

if args.dataset_name== 'VOC0712':
    annopath = os.path.join(VOCroot, 'VOC2007', 'Annotations', '%s.xml')
    imgpath = os.path.join(VOCroot, 'VOC2007', 'JPEGImages', '%s.jpg')
    imgsetpath = os.path.join(VOCroot, 'VOC2007', 'ImageSets', 'Main', '{:s}.txt')
    YEAR = '2007'
    devkit_path = VOCroot + 'VOC' + YEAR
    labelmap = VOC_CLASSES
elif args.dataset_name == 'VID2017':
    annopath = os.path.join(VIDroot, 'Annotations', 'VID', set_type, '%s.xml')
    imgpath = os.path.join(VIDroot, 'Data', 'VID', set_type, '%s.JPEG')
    imgsetpath = os.path.join(VIDroot, 'ImageSets', 'VID', '{:s}.txt')
    devkit_path = VIDroot[:-1]
    labelmap = VID_CLASSES

dataset_mean = (104, 117, 123)
ssd_dim = args.ssd_dim
pkl_dir = os.path.join(args.save_folder, args.model_dir.split('/')[-1])

if '.pth' in args.model_dir:
    trained_model = args.model_dir
elif 'VIDDET' in args.model_dir.split('/')[-1]:
    trained_model = os.path.join(args.model_dir, 'ssd300_VIDDET_' + args.literation +'.pth')
else:
    if args.tssd in ['tblstm',]:
        trained_model = os.path.join(args.model_dir, args.model_name+'_' + 'seq'+ args.dataset_name +'_'+ args.literation +'.pth')
    else:
        trained_model = os.path.join(args.model_dir, args.model_name+'_' + args.dataset_name +'_'+ args.literation +'.pth')

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        if args.dataset_name == 'VOC0712':
            obj_struct['pose'] = obj.find('pose').text
            obj_struct['truncated'] = int(obj.find('truncated').text)
            obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        objects.append(obj_struct)

    return objects


def get_output_dir(name, phase):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    filedir = os.path.join(name, phase)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir


def get_voc_results_file_template(image_set, cls):
    # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
    filename = 'det_' + image_set + '_%s.txt' % (cls)
    filedir = os.path.join(devkit_path, 'results')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path


def write_voc_results_file(all_boxes, dataset):
    for cls_ind, cls in enumerate(labelmap):
        # print('Writing {:s} VOC results file'.format(cls))
        filename = get_voc_results_file_template(set_type, cls)
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(dataset.ids):
                dets = all_boxes[cls_ind+1][im_ind]
                if dets == []:
                    continue
                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index[1], dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))


def do_python_eval(output_dir='output', use_07=True):
    cachedir = os.path.join(devkit_path, 'annotations_cache')
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = use_07
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for i, cls in enumerate(labelmap):
        filename = get_voc_results_file_template(set_type, cls)
        rec, prec, ap = voc_eval(
           filename, annopath, imgsetpath.format(args.set_file_name), cls, cachedir,
           ovthresh=0.5, use_07_metric=use_07_metric)
        aps += [ap]
        rec = [rec] if isinstance(rec, float) else rec
        prec = [prec] if isinstance(prec, float) else prec
        # rec_top = rec[args.top_k-1] if len(rec) > args.top_k else rec[-1]
        # prec_top = prec[args.top_k - 1] if len(prec) > args.top_k else prec[-1]
        print('{} AP = {:.4f}, Rec = {:.4f}, Prec = {:.4f}'.format(VID_CLASSES_name[VID_CLASSES.index(cls)], ap, rec[-1], np.max(prec)))
        with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
            pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    # print('~~~~~~~~')
    # print('Results:')
    # for ap in aps:
    #     print('{:.3f}'.format(ap))
    # print('{:.3f}'.format(np.mean(aps)))
    # print('~~~~~~~~')
    # print('')
    # print('--------------------------------------------------------------')
    # print('Results computed with the **unofficial** Python eval code.')
    # print('Results should be very close to the official MATLAB eval code.')
    # print('--------------------------------------------------------------')


def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=True):
    """rec, prec, ap = voc_eval(detpath,
                           annopath,
                           imagesetfile,
                           classname,
                           [ovthresh],
                           [use_07_metric])
Top level function that does the PASCAL VOC evaluation.
detpath: Path to detections
   detpath.format(classname) should produce the detection results file.
annopath: Path to annotations
   annopath.format(imagename) should be the xml annotations file.
imagesetfile: Text file containing the list of images, one image per line.
classname: Category name (duh)
cachedir: Directory for caching the annotations
[ovthresh]: Overlap threshold (default = 0.5)
[use_07_metric]: Whether to use VOC07's 11 point AP computation
   (default False)
"""
# assumes detections are in detpath.format(classname)
# assumes annotations are in annopath.format(imagename)
# assumes imagesetfile is a text file with each line an image name
# cachedir caches the annotations in a pickle file
# first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        if args.dataset_name == 'VID2017':
            lines_tmp = f.readlines()
            lines = []
            for i in range(len(lines_tmp)):
                lines.append(lines_tmp[i].split(' ')[0])
        else:
            lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath % (imagename))
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                   i + 1, len(imagenames)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # from load cachefile load gt, e.g.
        #{'ILSVRC2015_val_00004000/000010': [{'bbox': [241, 114, 451, 233], 'name': 'n02121808'},
        #                                   {'bbox': [106, 59, 384, 245], 'name': 'n02484322'}],
        #'ILSVRC2015_val_00000000/000000': [{'bbox': [416, 6, 605, 171], 'name': 'n01662784'}],
        #'ILSVRC2015_val_00000001/000158': [{'bbox': [644, 137, 903, 384], 'name': 'n01662784'}]}
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0 # the total number of gt objects in a class
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        #e.g. R = [{'name': 'n02121808', 'bbox': [241, 114, 451, 233]}]
        bbox = np.array([x['bbox'] for x in R])
        if args.dataset_name == 'VOC0712':
            difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        else:
            difficult = np.array([False for _ in R]).astype(np.bool)
        det = [False] * len(R) # False : have not been detected
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    #e.g.: /home/sean/data/ILSVRC/results/det_val_n02834778.txt
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:
        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence: greater->smaller
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids) # the total number of det objects in a class
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd): # each det boxes are compared with all
            R = class_recs[image_ids[d]]  # gt
            bb = BB[d, :].astype(float) # det
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float) # gt
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp) # compute top k
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        # original latter all = -1.
        rec = 0.
        prec = 0.
        ap = 0.

    return rec, prec, ap


def test_net(save_folder, net, cuda, dataset, transform, top_k,
             im_size=300, thresh=0.05, tssd='ssd'):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(dataset)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap)+1)]

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    all_time = 0.
    output_dir = get_output_dir(pkl_dir, args.literation+'_'+args.dataset_name+'_'+ args.set_file_name+'_tub'+str(args.tub)+'_'+str(args.tub_thresh)+'_'+str(args.tub_generate_score))
    det_file = os.path.join(output_dir, 'detections.pkl')
    state = [None] * 6 if tssd in ['lstm', 'edlstm', 'tblstm','tbedlstm', 'gru', 'outlstm'] else None
    pre_video_name = None
    for i in range(num_images):
        im, gt, h, w, _ = dataset.pull_item(i)
        # if len(gt) == 0:
        #     continue
        img_id = dataset.pull_img_id(i)
        # print(img_id[1])
        video_name = img_id[1].split('/')[0]
        if video_name != pre_video_name:
            state = [None] * 6 if tssd in['lstm', 'edlstm','tblstm', 'tbedlstm', 'gru', 'outlstm'] else None
            init_tub = True
            pre_video_name = video_name
        else:
            init_tub = False

        x = Variable(im.unsqueeze(0))
        if args.cuda:
            x = x.cuda()
        if tssd == 'ssd':
            _t['im_detect'].tic()
            detections,_ = net(x)
            detect_time = _t['im_detect'].toc(average=False)
            detections = detections.data
        else:
            _t['im_detect'].tic()
            detections, state, _ = net(x, state, init_tub)
            detect_time = _t['im_detect'].toc(average=False)
            detections = detections.data
        if i>10:
            all_time += detect_time

        # skip j = 0, because it's the background class
        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            mask = dets[:, 0].gt(0.).expand(dets.size(-1), dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, dets.size(-1))
            if dets.dim() == 0:
                continue
            boxes = dets[:, 1:-1] if dets.size(-1)==6 else dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            scores = dets[:, 0].cpu().numpy()
            cls_dets = np.hstack((boxes.cpu().numpy(), scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            all_boxes[j][i] = cls_dets
        if i % 100 == 0:
            print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1,
                                                    num_images, detect_time))
    print('all time:', all_time)
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    evaluate_detections(all_boxes, output_dir, dataset)


def evaluate_detections(box_list, output_dir, dataset):
    write_voc_results_file(box_list, dataset)
    do_python_eval(output_dir)


if __name__ == '__main__':
    # load net
    if args.dataset_name == 'VOC0712':
        num_classes = len(VOC_CLASSES) + 1 # +1 background
        dataset = VOCDetection(VOCroot, [('2007', set_type)], BaseTransform(ssd_dim, dataset_mean),
                               AnnotationTransform(dataset_name=args.dataset_name), dataset_name=args.dataset_name )
    elif args.dataset_name == 'VID2017':
        num_classes = len(VID_CLASSES) + 1 # +1 background
        dataset = VOCDetection(VIDroot, set_type, BaseTransform(ssd_dim, dataset_mean),
                               AnnotationTransform(dataset_name=args.dataset_name),
                               dataset_name=args.dataset_name, set_file_name=args.set_file_name)
    if args.detection:
        net = build_ssd('test', ssd_dim, num_classes, tssd=args.tssd,
                        top_k=args.top_k,
                        thresh=args.confidence_threshold,
                        nms_thresh=args.nms_threshold,
                        attention=args.attention, #o_ratio=args.oa_ratio[0], a_ratio=args.oa_ratio[1],
                        tub=args.tub,
                        tub_thresh=args.tub_thresh,
                        tub_generate_score=args.tub_generate_score)

        net.load_state_dict(torch.load(trained_model))
        net.eval()
        print('Finished loading model!', args.model_dir, args.literation,'tub='+str(args.tub), 'tub_thresh='+str(args.tub_thresh), 'tub_score='+str(args.tub_generate_score))
        # load data
        if args.cuda:
            net = net.cuda()
            cudnn.benchmark = True
        # evaluation
        test_net(args.save_folder, net, args.cuda, dataset,
                 BaseTransform(net.size, dataset_mean), args.top_k, ssd_dim,
                 thresh=args.confidence_threshold, tssd=args.tssd)
    else:
        out_dir = get_output_dir(pkl_dir, args.literation+'_'+args.dataset_name+'_'+ args.set_file_name)
        print('Without detection', out_dir)
        do_python_eval(out_dir)
    print('Finished!', args.model_dir, args.literation, 'tub='+str(args.tub), 'tub_thresh='+str(args.tub_thresh), 'tub_score='+str(args.tub_generate_score))

