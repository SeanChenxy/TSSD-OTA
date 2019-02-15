import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data import base_transform, VID_CLASSES, VID_CLASSES_name, MOT_CLASSES
from ssd import build_ssd
from layers.modules import  AttentionLoss
import os
import numpy as np
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
dataset_name = 'VID2017'
if dataset_name == 'VID2017':
    model_dir='/home/sean/Documents/ssd.pytorch/weights/VIDtssd/trained_model/ssd300_VID2017_6543.pth'
    video_name='/data/ILSVRC/Data/VID/snippets/val/ILSVRC2015_val_00007010.mp4'
    labelmap = VID_CLASSES
    num_classes = len(VID_CLASSES) + 1
    prior = 'v2'
    confidence_threshold = 0.75
    nms_threshold = 0.5
    top_k = 200
elif dataset_name == 'MOT15':
    model_dir='./weights/tssd300_MOT15_SAL222/ssd300_seqMOT15_4000.pth'
    val_list = ['TUD-Campus.mp4', 'ETH-Sunnyday.mp4', 'ETH-Pedcross2.mp4', 'ADL-Rundle-8.mp4', 'Venice-2.mp4', 'KITTI-17.mp4']
    all_list = {0:'ADL-Rundle-1.mp4', 1:'ADL-Rundle-3.mp4', 2:'ADL-Rundle-6.mp4', 3:'ADL-Rundle-8.mp4', 4:'AVG-TownCentre.mp4',
                5:'ETH-Bahnhof.mp4', 6:'ETH-Crossing.mp4', 7:'ETH-Jelmoli.mp4', 8:'ETH-Linthescher.mp4', 9:'ETH-Pedcross2.mp4',
                10:'ETH-Sunnyday.mp4', 11:'PETS09-S2L1.mp4', 12:'PETS09-S2L2.mp4', 13:'TUD-Campus.mp4', 14:'TUD-Crossing.mp4',
                15:'TUD-Stadtmitte.mp4', 16:'Venice-1.mp4', 17:'Venice-2.mp4'}

    video_name = '/data/MOT/snippets/'+all_list[15]

    labelmap = MOT_CLASSES
    num_classes = len(MOT_CLASSES) + 1
    prior = 'v3'
    confidence_threshold = 0.1
    nms_threshold = 0.3
    top_k = 400

else:
    raise ValueError("dataset [%s] not recognized." % dataset_name)

model_name= 'ssd300'
ssd_dim=300
tssd = 'tblstm'
attention = True
tub = 10
tub_thresh = 1
tub_generate_score = 0.1

# save_dir = os.path.join('./demo/OTA', video_name.split('/')[-1].split('.')[0])
save_dir = None
if save_dir and not os.path.exists(save_dir):
        os.mkdir(save_dir)

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
                    tub_generate_score=tub_generate_score)
    net.load_state_dict(torch.load(trained_model))
    net.eval()

    print('Finished loading model!', model_dir)

    net = net.cuda()
    cudnn.benchmark = True

    frame_num = 0
    cap = cv2.VideoCapture(video_name)
    w, h = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(w, h)
    if save_dir:
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        size = (640, 480)
        record = cv2.VideoWriter(os.path.join(save_dir,video_name.split('/')[-1].split('.')[0]+'_OTA.avi'), fourcc, cap.get(cv2.CAP_PROP_FPS), size)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

    att_criterion = AttentionLoss((h, w))
    state = [None] * 6 if tssd in ['lstm', 'tblstm', 'outlstm'] else None
    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        frame_draw = frame.copy()
        frame_num += 1
        im_trans = base_transform(frame, ssd_dim, mean)
        x = Variable(torch.from_numpy(im_trans).unsqueeze(0).permute(0, 3, 1, 2), volatile=True)
        x = x.cuda()
        if tssd == 'ssd':
            detections, att_map = net(x)
            detections = detections.data
        else:
            detections, state, att_map = net(x, state)
            detections = detections.data
            # print(np.around(t_diff, decimals=4))
        out = list()
        for j in range(1, detections.size(1)):
            for k in range(detections.size(2)):
                dets = detections[0, j, k, :]
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

        if attention:
            _, up_attmap = att_criterion(att_map)  # scale, batch, tensor(1,h,w)
            att_target = up_attmap[0][0].cpu().data.numpy().transpose(1, 2, 0)
        for object in out:
            x_min, y_min, x_max, y_max, cls, score, identity = object
            if identity in [0]:
                color = (0, 0, 255)
            elif identity in [1]:
                color = (0, 200, 0)
            elif identity in [2]:
                color = (255, 128, 0)
            elif identity in [3]:
                color = (255, 0, 255)
            elif identity in [4]:
                color = (0, 128, 255)
            elif identity in [5]:
                color = (255, 128, 128)
            else:
                color = (255, 0, 0)
            cv2.rectangle(frame_draw, (x_min, y_min), (x_max, y_max), color, thickness=2)
            cv2.fillConvexPoly(frame_draw, np.array(
                [[x_min - 1, y_min], [x_min - 1, y_min - 50], [x_max + 1, y_min - 50], [x_max + 1, y_min]], np.int32),
                               color)
            if dataset_name == 'VID2017':
                put_str = str(int(identity))+':'+VID_CLASSES_name[cls] +':'+ str(np.around(score, decimals=2))
            else:
                put_str = str(int(identity))

            cv2.putText(frame_draw, put_str,
                        (x_min + 10, y_min - 10), cv2.FONT_HERSHEY_DUPLEX, 1, color=(255, 255, 255), thickness=1)
            print(str(frame_num) + ':' + str(np.around(score, decimals=2)) + ','+VID_CLASSES_name[cls])
        if not out:
            print(str(frame_num))
        cv2.imshow('frame', cv2.resize(frame_draw, (640,360)))
        if save_dir:
            frame_write = cv2.resize(frame_draw, size)
            record.write(frame_write)
        ch = cv2.waitKey(1)
        if ch == 32:
            while 1:
                in_ch = cv2.waitKey(10)
                if in_ch == 115: # 's'
                    if save_dir:
                        print('save: ', frame_num)
                        torch.save(out, os.path.join(save_dir, tssd+'_%s.pkl' % str(frame_num)))
                        cv2.imwrite(os.path.join(save_dir, '%s.jpg' % str(frame_num)), frame)
                elif in_ch == 32:
                    break

    cap.release()
    if save_dir:
        record.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

