import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data import base_transform, MOT_CLASSES
from ssd import build_ssd
import os
import numpy as np
import cv2
import time

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

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

data_size = {'MOT17-02':600, 'MOT17-04':1050, 'MOT17-05':837, 'MOT17-09':525, 'MOT17-10':654, 'MOT17-11': 900,
             'MOT17-13':750, 'MOT17-01':450, 'MOT17-03':1500, 'MOT17-06':1194, 'MOT17-07':500, 'MOT17-08':625, 'MOT17-12':900, 'MOT17-14':750,
             'ADL-Rundle-6': 525, 'ADL-Rundle-8': 654, 'ETH-Bahnhof': 1000, 'ETH-Pedcross2':837, 'ETH-Sunnyday': 354,
             'KITTI-13': 261, 'KITTI-17': 145, 'PETS09-S2L1':795, 'TUD-Campus':71,
             'TUD-Stadtmitte':179, 'Venice-2':600, 'ADL-Rundle-1':500, 'ADL-Rundle-3':625, 'AVG-TownCentre':450,
             'ETH-Crossing':219, 'ETH-Jelmoli':440, 'ETH-Linthescher':1194, 'KITTI-16':209, 'KITTI-19':1059,
             'PETS09-S2L2':436, 'TUD-Crossing':201, 'Venice-1':450
             }

model_dir='./weights/tssd300_MOT15_SAL222/ssd300_seqMOT15_4000.pth'
data_path = '/home/sean/data/MOT/MOT17Det/train/'

labelmap = MOT_CLASSES
num_classes = len(MOT_CLASSES) + 1
prior = 'v3'

model_name= 'ssd300'
confidence_threshold=0.3
nms_threshold =0.3
top_k=400
ssd_dim=300
vis = False
if model_dir.split('/')[2].split('_')[0][0]=='t':
    tssd = 'tblstm'
    attention = True
else:
    tssd = 'ssd'
    attention = False

refine = False
tub = 10
tub_thresh = 1
tub_generate_score = 0.3
tub_flag = '_t'+str(tub)+'s'+str(tub_thresh)+'g'+str(tub_generate_score)+'_nounique'

set_name = '2DMOT2015'
if set_name == '2DMOT2015':
    val_list = ['TUD-Campus', 'ETH-Sunnyday', 'ETH-Pedcross2', 'ADL-Rundle-8', 'Venice-2', 'KITTI-17']
    # val_list = ['ADL-Rundle-1' 'ADL-Rundle-3', 'AVG-TownCentre', 'ETH-Crossing', 'ETH-Jelmoli', 'ETH-Linthescher', 'KITTI-16', 'KITTI-19',
    #             'PETS09-S2L2', 'TUD-Crossing', 'Venice-1']
elif set_name == 'MOT17Det':
    val_list = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10', 'MOT17-11', 'MOT17-13']
    # val_list = ['MOT17-01', 'MOT17-03', 'MOT17-06', 'MOT17-07', 'MOT17-08', 'MOT17-12', 'MOT17-14']

output_flag = True
output_dir = '/home/sean/data/MOT/motchallenge-devkit/motchallenge/res/%s/%s' % (set_name, model_dir.split('/')[2] + tub_flag)
if not os.path.exists(output_dir) and output_flag:
    os.mkdir(output_dir)

torch.set_default_tensor_type('torch.cuda.FloatTensor')

def main():
    mean = (104, 117, 123)
    trained_model = model_dir

    print('loading model!')
    net = build_ssd('test', ssd_dim, num_classes, tssd=tssd,
                    top_k=top_k,
                    thresh=confidence_threshold,
                    nms_thresh=nms_threshold,
                    attention=attention,
                    prior=prior,
                    tub = tub,
                    tub_thresh = tub_thresh,
                    tub_generate_score=tub_generate_score,
                    bn=False)
    net.load_state_dict(torch.load(trained_model))
    net.eval()

    print('Finished loading model!', model_dir)

    net = net.cuda()
    cudnn.benchmark = True
    _t = {'im_detect': Timer(), 'misc': Timer()}
    all_time = 0.
    total_frame = 0

    for val in val_list:
        img_path = '/home/sean/data/MOT/%s/train/%s/img1' % (set_name, val)
        output_path = os.path.join(output_dir, val + '.txt')
        if output_flag:
            wf = open(output_path, 'w')
        frame_num = 0
        pre_frame = cv2.imread(os.path.join(img_path, '000001.jpg'))
        h, w, _ = pre_frame.shape
        state = [None] * 6 if tssd in ['lstm', 'tblstm'] else None
        init_tub = True

        for i in range(1, data_size[val]+1):
            frame = cv2.imread(os.path.join(img_path, str(i).zfill(6)+'.jpg'))

            frame_draw = frame.copy()
            frame_num += 1
            im_trans = base_transform(frame, ssd_dim, mean)
            x = Variable(torch.from_numpy(im_trans).unsqueeze(0).permute(0, 3, 1, 2), volatile=True)
            x = x.cuda()
            if tssd == 'ssd':
                _t['im_detect'].tic()
                detections, _ = net(x)
                detect_time = _t['im_detect'].toc(average=False)
                detections = detections.data
            else:
                _t['im_detect'].tic()
                detections, state, _ = net(x, state, init_tub)
                detect_time = _t['im_detect'].toc(average=False)
                detections = detections.data
                init_tub = False
            all_time += detect_time
            out = list()
            for j in range(1, detections.size(1)):
                for k in range(detections.size(2)):
                    dets = detections[0, j, k, :]
                    # mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
                    # dets = torch.masked_select(dets, mask).view(-1, 5)
                    if dets.dim() == 0:
                        continue

                    boxes = dets[1:-1] if dets.size(0) == 6 else dets[1:]
                    identity = dets[-1] if dets.size(0) == 6 else -1
                    x_min = int(boxes[0] * w)
                    x_max = int(boxes[2] * w)
                    y_min = int(boxes[1] * h)
                    y_max = int(boxes[3] * h)

                    score = dets[0]
                    if score > confidence_threshold:
                        out.append([x_min, y_min, x_max, y_max, j - 1, score, identity])
                        wf.write(str(frame_num)+','+str(int(identity))+','+str(x_min)+','+str(y_min)+','+str(x_max-x_min)+','+str(y_max-y_min)+','+str(np.around(score, decimals=2))+',-1,-1,-1\n')
            print(val + ':' + str(frame_num))

            if vis:
                for object in out:
                    color = (0, 0, 255)
                    x_min, y_min, x_max, y_max, cls, score, identity = object
                    cv2.rectangle(frame_draw, (x_min, y_min), (x_max, y_max), color, thickness=2)
                    cv2.fillConvexPoly(frame_draw, np.array(
                        [[x_min - 1, y_min], [x_min - 1, y_min - 50], [x_max + 1, y_min - 50], [x_max + 1, y_min]], np.int32),
                                       color)

                    put_str = str(int(identity))+':'+ str(np.around(score, decimals=2))

                    cv2.putText(frame_draw, put_str,
                                (x_min + 10, y_min - 10), cv2.FONT_HERSHEY_DUPLEX, 1, color=(255, 255, 255), thickness=1)
                cv2.imshow('frame', cv2.resize(frame_draw, (640,360)))
                ch = cv2.waitKey(1)
                if ch == 32:
                    while 1:
                        in_ch = cv2.waitKey(10)
                        if in_ch == 32:
                            break

        if output_flag:
            wf.close()
        total_frame += frame_num
    fps = total_frame/all_time
    print('frames:', total_frame, 'all time:', all_time, 'fps:', fps)
    print(output_dir.split('/')[-1])

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

