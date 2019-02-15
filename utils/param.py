import torch
from ssd import build_ssd

ssd_net = build_ssd('train', 300, 31, tssd='lstm', attention=False)

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

print_network(ssd_net)
