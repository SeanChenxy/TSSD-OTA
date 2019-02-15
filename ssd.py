import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import v2, v3
from layers import half_decode
from data import v2 as cfg
import os

class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, base, extras, head, num_classes, top_k=200, thresh=0.01, nms_thresh=0.45, attention=False, prior='v2'):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.attention_flag = attention
        # TODO: implement __call__ in PriorBox
        if prior=='v2':
            self.priorbox = PriorBox(v2)
        elif prior=='v3':
            self.priorbox = PriorBox(v3)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = 300

        # SSD network
        self.vgg = nn.ModuleList(base)
        self.conv4_3_layer = (23, 33)[len(self.vgg)>40]
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)
        self.extras_skip = (2, 3)[len(self.vgg)>40]
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        if self.attention_flag:
            self.attention = nn.ModuleList([ConvAttention(512),ConvAttention(256)])
                                            # ConvAttention(512),ConvAttention(256),
                                            # ConvAttention(256),ConvAttention(256)])
            print(self.attention)
        if phase == 'test':
            self.softmax = nn.Softmax()
            self.detect = Detect(num_classes, 0, top_k=top_k, conf_thresh=thresh, nms_thresh=nms_thresh)
                                # num_classes, bkg_label, top_k, conf_thresh, nms_thresh

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3*batch,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()
        a_map = list()

        # apply vgg up to conv4_3 relu
        for k in range(self.conv4_3_layer):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(self.conv4_3_layer, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % self.extras_skip == 1:
                sources.append(x)

        # apply multibox head to source layers
        if self.attention_flag:
            for i, (x, l, c) in enumerate(zip(sources, self.loc, self.conf)):
                a_map.append(self.attention[i//3](x))
                a_feat = x*a_map[-1]
                loc.append(l(a_feat).permute(0, 2, 3, 1).contiguous())  # [ith_multi_layer, batch, height, width, out_channel]
                conf.append(c(a_feat).permute(0, 2, 3, 1).contiguous())
        else:
            for (x, l, c) in zip(sources, self.loc, self.conf):
                loc.append(l(x).permute(0, 2, 3, 1).contiguous()) # [ith_multi_layer, batch, height, width, out_channel]
                conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(-1, self.num_classes)),  # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors,
            )
        return output, tuple(a_map)

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

class ConvAttention(nn.Module):

    def __init__(self, inchannel):
        super(ConvAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(inchannel, inchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            # nn.ConvTranspose2d(int(inchannel/2), int(inchannel/4), kernel_size=3, stride=2, padding=1, output_padding=0, bias=False),
            nn.Conv2d(inchannel, inchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(inchannel, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, feats):
        return self.attention(feats)



# https://www.jianshu.com/p/72124b007f7d
class ConvLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, input_size, hidden_size, phase='train'):
        super(ConvLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, 3, padding=1)
        self.phase = phase

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                Variable(torch.zeros(state_size), volatile=(False, True)[self.phase=='test']),
                Variable(torch.zeros(state_size), volatile=(False, True)[self.phase=='test'])
            )

        prev_cell, prev_hidden = prev_state
        # prev_hidden_drop = F.dropout(prev_hidden, training=(False, True)[self.phase=='train'])
        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((F.dropout(input_, p=0.2, training=(False,True)[self.phase=='train']), prev_hidden), 1)
        # stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = F.sigmoid(in_gate)
        remember_gate = F.sigmoid(remember_gate)
        out_gate = F.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = F.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * F.tanh(cell)

        return cell, hidden

    def init_state(self, input_):
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]
        state_size = [batch_size, self.hidden_size] + list(spatial_size)
        state = (
            Variable(torch.zeros(state_size), volatile=(False, True)[self.phase == 'test']),
            Variable(torch.zeros(state_size), volatile=(False, True)[self.phase == 'test'])
        )
        return state

class ConvGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size=3, cuda_flag=True, phase='train'):
        super(ConvGRUCell, self).__init__()
        self.input_size = input_size
        self.cuda_flag = cuda_flag
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.ConvGates = nn.Conv2d(self.input_size + self.hidden_size, 2 * self.hidden_size, 3,
                                   padding=self.kernel_size // 2)
        self.Conv_ct = nn.Conv2d(self.input_size + self.hidden_size, self.hidden_size, 3, padding=self.kernel_size // 2)
        dtype = torch.FloatTensor
        self.phase = phase

    def forward(self, input, hidden):
        if hidden is None:
            size_h = [input.data.size()[0], self.hidden_size] + list(input.data.size()[2:])
            if self.cuda_flag == True:
                hidden = (Variable(torch.zeros(size_h), volatile=(False, True)[self.phase=='test']).cuda(), )
            else:
                hidden = (Variable(torch.zeros(size_h), volatile=(False, True)[self.phase=='test']), )
        hidden = hidden[-1]
        c1 = self.ConvGates(torch.cat((F.dropout(input,p=0.2,training=(False,True)[self.phase=='train']), hidden), 1))
        (rt, ut) = c1.chunk(2, 1)
        reset_gate = F.sigmoid(rt)
        update_gate = F.sigmoid(ut)
        gated_hidden = torch.mul(reset_gate, hidden)
        p1 = self.Conv_ct(torch.cat((input, gated_hidden), 1))
        ct = F.tanh(p1)
        next_h = torch.mul(update_gate, hidden) + (1 - update_gate) * ct
        return (next_h, )

    def init_state(self, input):
        size_h = [input.data.size()[0], self.hidden_size] + list(input.data.size()[2:])
        if self.cuda_flag == True:
            hidden = (Variable(torch.zeros(size_h), volatile=(False, True)[self.phase == 'test']).cuda(),)
        else:
            hidden = (Variable(torch.zeros(size_h), volatile=(False, True)[self.phase == 'test']),)
        return hidden

class TSSD(nn.Module):

    def __init__(self, phase, base, extras, head, num_classes, lstm='lstm', size=300,
                 top_k=200,thresh= 0.01,nms_thresh=0.45, attention=False, prior='v2',
                 tub=0, tub_thresh=1.0, tub_generate_score=0.7):
        super(TSSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        if prior=='v2':
            self.priorbox = PriorBox(v2)
        elif prior=='v3':
            self.priorbox = PriorBox(v3)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size
        self.attention_flag = attention

        # SSD network
        self.vgg = nn.ModuleList(base)
        self.conv4_3_layer = (23, 33)[len(self.vgg)>40]
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)
        self.extras_skip = (2, 3)[len(self.vgg)>40]
        self.lstm_mode = lstm

        self.rnn = nn.ModuleList(head[2])
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        print(self.rnn)
        if self.attention_flag:
            in_channel = 512
            self.attention = nn.ModuleList([ConvAttention(in_channel*2), ConvAttention(in_channel)])
            print(self.attention)
        if phase == 'test':
            self.tub = tub
            self.softmax = nn.Softmax()
            self.detect = Detect(num_classes, 0, top_k=top_k, conf_thresh=thresh, nms_thresh=nms_thresh,
                                 tub=tub, tub_thresh=tub_thresh, tub_generate_score=tub_generate_score)

    def forward(self, tx, state=None, init_tub=False):
        if self.phase == "train":
            rnn_state = [None] * 6
            seq_output = list()
            seq_sources = list()
            seq_a_map = []
            for time_step in range(tx.size(1)):
                x = tx[:,time_step]
                sources = list()
                loc = list()
                conf = list()
                a_map = list()

                # apply vgg up to conv4_3 relu
                for k in range(self.conv4_3_layer):
                    x = self.vgg[k](x)

                s = self.L2Norm(x)
                sources.append(s)

                # apply vgg up to fc7
                for k in range(23, len(self.vgg)):
                    x = self.vgg[k](x)
                sources.append(x)

                # apply extra layers and cache source layer outputs
                for k, v in enumerate(self.extras):
                    x = F.relu(v(x), inplace=True)
                    if k % self.extras_skip == 1:
                        sources.append(x)
                seq_sources.append(sources)
                # apply multibox head to source layers
                if self.attention_flag:
                   for i, (x, l, c) in enumerate(zip(sources, self.loc, self.conf)):
                        if time_step == 0:
                            rnn_state[i] = self.rnn[i // 3].init_state(x)
                        a_map.append(self.attention[i//3](torch.cat((x, rnn_state[i][-1]),1)))
                        a_feat =  x *a_map[-1]
                        rnn_state[i] = self.rnn[i//3](a_feat, rnn_state[i])
                        conf.append(c(rnn_state[i][-1]).permute(0, 2, 3, 1).contiguous())
                        loc.append(l(rnn_state[i][-1]).permute(0, 2, 3, 1).contiguous())
                else:
                    for i, (x, l, c) in enumerate(zip(sources, self.loc, self.conf)):
                        rnn_state[i] = self.rnn[i//3](x, rnn_state[i])
                        conf.append(c(rnn_state[i][-1]).permute(0, 2, 3, 1).contiguous())
                        loc.append(l(rnn_state[i][-1]).permute(0, 2, 3, 1).contiguous())

                loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
                conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

                output = (
                    loc.view(loc.size(0), -1, 4),
                    conf.view(conf.size(0), -1, self.num_classes),
                    self.priors,
                )
                seq_output.append(output)
                seq_a_map.append(tuple(a_map))

            return tuple(seq_output), tuple(seq_a_map)
        elif self.phase == 'test':

            sources = list()
            loc = list()
            conf = list()
            a_map = list()

            # apply vgg up to conv4_3 relu
            for k in range(self.conv4_3_layer):
                tx = self.vgg[k](tx)

            s = self.L2Norm(tx)
            sources.append(s)

            # apply vgg up to fc7
            for k in range(23, len(self.vgg)):
                tx = self.vgg[k](tx)
            sources.append(tx)

            # apply extra layers and cache source layer outputs
            for k, v in enumerate(self.extras):
                tx = F.relu(v(tx), inplace=True)
                if k % self.extras_skip == 1:
                    sources.append(tx)

            # apply multibox head to source layers
            if self.attention_flag:
                for i, (x, l, c) in enumerate(zip(sources, self.loc, self.conf)):
                    if state[i] is None:
                        state[i] = self.rnn[i // 3].init_state(x)
                    a_map.append(self.attention[i // 3](torch.cat((x, state[i][-1]), 1)))
                    a_feat = x * a_map[-1]
                    state[i] = self.rnn[i // 3](a_feat, state[i])
                    conf.append(c(state[i][-1]).permute(0, 2, 3, 1).contiguous())
                    loc.append(l(state[i][-1]).permute(0, 2, 3, 1).contiguous())
            else:
                for i, (x, l, c) in enumerate(zip(sources, self.loc, self.conf)):
                    state[i] = self.rnn[i//3](x, state[i])
                    conf.append(c(state[i][-1]).permute(0, 2, 3, 1).contiguous())
                    loc.append(l(state[i][-1]).permute(0, 2, 3, 1).contiguous())

            loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
            conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
            if self.tub:
                for a_idx, a in enumerate(a_map[:3]):
                    if not a_idx:
                        tub_tensor = a
                        tub_tensor_size = a.size()[2:]
                    else:
                        tub_tensor = torch.cat((tub_tensor, F.upsample(a, tub_tensor_size, mode='bilinear')), dim=1)
                if init_tub:
                    self.detect.init_tubelets()
                output = self.detect(
                    loc.view(loc.size(0), -1, 4),  # loc preds
                    self.softmax(conf.view(-1, self.num_classes)),  # conf preds
                    self.priors.type(type(tx.data)),  # default boxes
                    tub_tensor
                )
            else:
                output = self.detect(
                    loc.view(loc.size(0), -1, 4),  # loc preds
                    self.softmax(conf.view(-1, self.num_classes)),  # conf preds
                    self.priors.type(type(tx.data)),  # default boxes
                )

            return output, state, tuple(a_map)


    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    # conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    conv7 = nn.Conv2d(1024, 512, kernel_size=1)
    if batch_norm:
        layers += [pool5, conv6, nn.BatchNorm2d(1024),
                   nn.ReLU(inplace=True), conv7, nn.BatchNorm2d(512), nn.ReLU(inplace=True)]
    else:
        layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                if batch_norm:
                    layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1), nn.BatchNorm2d(cfg[k + 1])]
                else:
                    layers += [nn.Conv2d(in_channels, cfg[k + 1],
                                         kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                if batch_norm and k in [7]:
                    layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag]), nn.BatchNorm2d(v)]

                else:
                    layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers

def multibox(vgg, extra_layers, cfg, num_classes, lstm=None, phase='train', batch_norm=False):
    loc_layers = []
    conf_layers = []
    rnn_layer = []
    vgg_source = ([24, -2], [34, -3])[batch_norm==True]
    for k, v in enumerate(vgg_source):

        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    key_extra_layers = (extra_layers[1::2], extra_layers[1::3])[batch_norm==True]
    for k, v in enumerate(key_extra_layers, 2):

        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    if lstm in ['tblstm']:
        rnn_layer = [ConvLSTMCell(512,512,phase=phase), ConvLSTMCell(256,256,phase=phase)]
    elif lstm in ['gru']:
        rnn_layer = [ConvGRUCell(512,512,phase=phase), ConvGRUCell(256,256,phase=phase)]
    return vgg, extra_layers, (loc_layers, conf_layers, rnn_layer)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512], # output channel
    '512': [],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    # '300': [256, 'S', 512, 256, 'S', 512, 256, 512, 256, 512],
    '512': [],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
   # '300': [5, 5, 5, 5, 5, 5],
    # '300': [4, 4, 4, 4, 4, 4],
    '512': [],
}


def build_ssd(phase, size=300, num_classes=21, tssd='ssd', top_k=200, thresh=0.01, prior='v2', bn=False,
              nms_thresh=0.45, attention=False, tub=0, tub_thresh=1.0, tub_generate_score=0.7):
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return
    if size != 300:
        print("Error: Sorry only SSD300 is supported currently!")
        return
    if tssd == 'ssd':
        return SSD(phase, *multibox(vgg(base[str(size)], 3, batch_norm=bn),
                                    add_extras(extras[str(size)], 512, batch_norm=bn),
                                    mbox[str(size)], num_classes, phase=phase, batch_norm=bn), num_classes,
                   top_k=top_k,thresh= thresh,nms_thresh=nms_thresh, attention=attention, prior=prior)
    else:
        return TSSD(phase, *multibox(vgg(base[str(size)], 3, batch_norm=bn),
                                add_extras(extras[str(size)], 512),
                                mbox[str(size)], num_classes, lstm=tssd, phase=phase,batch_norm=bn),
                    num_classes, lstm=tssd, size=size, top_k=top_k, thresh=thresh, prior=prior,
                    nms_thresh=nms_thresh, attention=attention,
                    tub=tub, tub_thresh=tub_thresh, tub_generate_score=tub_generate_score
                    )
