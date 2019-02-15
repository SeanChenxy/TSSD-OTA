import torch
import torch.nn as nn
from torch.autograd import Variable


class AttentionLoss(nn.Module):
    def __init__(self, dim, cuda=True):
        super(AttentionLoss, self).__init__()
        self.criterion = nn.BCELoss().cuda() if cuda else nn.BCEWithLogitsLoss()
        if isinstance(dim, tuple):
            self.upsample = nn.Upsample(dim, mode='bilinear')
        else:
            self.upsample = nn.Upsample((dim,dim), mode='bilinear')
        self.cuda = cuda

    def forward(self, att_map, masks=None, viz=False):

        loss_att = 0
        if isinstance(att_map[0], tuple):
            seq_upsampled_att_map = []
            for time_step, seq_att_map in enumerate(att_map):
                upsampled_att_map = []
                for scale_att in seq_att_map:
                    upsampled_att_map.append(self.upsample(scale_att))
                    if masks is not None:
                        loss_att += self.criterion(upsampled_att_map[-1], masks[:,time_step,:,:])
                seq_upsampled_att_map.append(upsampled_att_map)
            return loss_att/len(att_map), seq_upsampled_att_map
        else:
            upsampled_att_map = []
            for scale_att in att_map:
                upsampled_att_map.append(self.upsample(scale_att))
                if masks is not None:
                    loss_att += self.criterion(upsampled_att_map[-1], masks)
            return loss_att, upsampled_att_map
