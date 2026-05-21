import ltn
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules import Module


# * Utils methods: convolutional block definition (OK)
def conv_block(in_channels, out_channels):
    '''
    returns a block conv-bn-relu-pool
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class ProtoNet(nn.Module):
    '''
    Model as described in the reference paper,
    source: https://github.com/jakesnell/prototypical-networks/blob/f0c48808e496989d01db59f86d4449d7aee9ab0c/protonets/models/few_shot.py#L62-L84
    '''
    # & Using the convolutional blocks to build the prototypical network (OK)
    def __init__(self, x_dim=1, hid_dim=64, z_dim=64):
        super(ProtoNet, self).__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )

    # & Compute the embeddings for support and query points to use during episodes (OK)
    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)



class LearnableProtoNet_CNN(torch.nn.Module):
    def __init__(self, num_classes, z_dim=64):
        super().__init__()
        # Embedding network
        self.embedding = ProtoNet(z_dim=z_dim)

        # NOTE(corr-14): the original line was
        #     self.prototypes = nn.Parameter(torch.randn(...)).to(ltn.device)
        # but `Parameter.to(device)` returns a plain Tensor (not a Parameter)
        # whenever the device move is not a no-op. On GPU this silently
        # demoted the result so it was never registered in self._parameters
        # and never updated by the optimizer. Letting the surrounding module
        # `.to(device)` call move the parameter preserves the Parameter type.
        self.prototypes = torch.nn.Parameter(torch.randn(num_classes, z_dim))

    def forward(self, x):
        x = self.embedding(x)
        return x