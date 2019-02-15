# config.py
import os.path

# gets home dir cross platform
home = os.path.expanduser("/")
vocdir = os.path.join(home,"data/VOCdevkit/")
viddir = os.path.join(home,"data/ILSVRC/")
mot17detdir = os.path.join(home,"data/MOT/MOT17Det/")
mot15dir = os.path.join(home,"data/MOT/2DMOT2015/")

# note: if you used our download scripts, this should be right
VOCroot = vocdir # path to VOCdevkit root dir
VIDroot = viddir
MOT17Detroot = mot17detdir
MOT15root = mot15dir

# default batch size
BATCHES = 32
# data reshuffled at every epoch
SHUFFLE = True
# number of subprocesses to use for data loading
WORKERS = 4


#SSD300 CONFIGS
# newer version: use additional conv11_2 layer as last layer before multibox layers
v2 = {
    'feature_maps' : [38, 19, 10, 5, 3, 1],

    'min_dim' : 300,

    'steps' : [8, 16, 32, 64, 100, 300],

    'min_sizes' : [30, 60, 111, 162, 213, 264],

    'max_sizes' : [60, 111, 162, 213, 264, 315],

    # 'aspect_ratios' : [[2, 1/2], [2, 1/2, 3, 1/3], [2, 1/2, 3, 1/3],
    #                    [2, 1/2, 3, 1/3], [2, 1/2], [2, 1/2]],
    'aspect_ratios' : [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    # 'aspect_ratios' : [[2,3], [2, 3], [2, 3], [2, 3], [2,3], [2,3]],

    'variance' : [0.1, 0.2],

    'clip' : True,

    'flip': True,

    'name' : 'v2',
}

v3 = {
    'feature_maps' : [38, 19, 10, 5, 3, 1],

    'min_dim' : 300,

    'steps' : [8, 16, 32, 64, 100, 300],

    'min_sizes' : [30, 60, 111, 162, 213, 264],

    'max_sizes' : [60, 111, 162, 213, 264, 315],

    # 'aspect_ratios': [[1 / 2, 1 / 3], [1 / 2, 1 / 3], [1 / 2, 1 / 3], [1 / 2, 1 / 3],
    #                   [1 / 2, 1 / 3], [1 / 2, 1 / 3]],

    # 'aspect_ratios' : [[2,3,4], [2, 3,4], [2, 3, 4], [2, 3, 4], [2,3,4], [2,3,4]],
    'aspect_ratios': [[1 / 2, 1 / 3, 1/4], [1 / 2, 1 / 3, 1/4], [1 / 2, 1 / 3, 1/4], [1 / 2, 1 / 3, 1/4], [1 / 2, 1 / 3, 1/4], [1 / 2, 1 / 3, 1/4]],

    'variance' : [0.1, 0.2],

    'clip' : True,

    'flip': False,

    'name' : 'v3',
}

# use average pooling layer as last layer before multibox layers
v1 = {
    'feature_maps' : [38, 19, 10, 5, 3, 1],

    'min_dim' : 300,

    'steps' : [8, 16, 32, 64, 100, 300],

    'min_sizes' : [30, 60, 114, 168, 222, 276],

    'max_sizes' : [-1, 114, 168, 222, 276, 330],

    # 'aspect_ratios' : [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
    'aspect_ratios' : [[1,1,2,1/2],[1,1,2,1/2,3,1/3],[1,1,2,1/2,3,1/3],
                        [1,1,2,1/2,3,1/3],[1,1,2,1/2,3,1/3],[1,1,2,1/2,3,1/3]],

    'variance' : [0.1, 0.2],

    'clip' : True,

    'name' : 'v1',
}
