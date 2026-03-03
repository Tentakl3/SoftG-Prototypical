import torch
from torch.nn.init import xavier_uniform_, normal_, kaiming_uniform_
from z3 import *

# we define predicate digit
class MNISTConv(torch.nn.Module):
    def __init__(self, conv_channels_sizes=(1, 6, 16), kernel_sizes=(5, 5), linear_layers_sizes=(256, 100)):
        super(MNISTConv, self).__init__()
        self.conv_layers = torch.nn.ModuleList([torch.nn.Conv2d(conv_channels_sizes[i - 1], conv_channels_sizes[i],
                                                                kernel_sizes[i - 1])
                                                  for i in range(1, len(conv_channels_sizes))])
        self.relu = torch.nn.ReLU()  # relu is used as activation for the conv layers
        self.tanh = torch.nn.Tanh()  # tanh is used as activation for the linear layers
        self.maxpool = torch.nn.MaxPool2d((2, 2))
        self.linear_layers = torch.nn.ModuleList([torch.nn.Linear(linear_layers_sizes[i - 1], linear_layers_sizes[i])
                                                  for i in range(1, len(linear_layers_sizes))])
        self.batch_norm_layers = torch.nn.ModuleList([torch.nn.BatchNorm1d(linear_layers_sizes[i])
                                                      for i in range(1, len(linear_layers_sizes))])

        self.softmax = torch.nn.Softmax(dim=-1)

        self.init_weights()

    def forward(self, x):
        for conv in self.conv_layers:
            x = self.relu(conv(x))
            x = self.maxpool(x)
        x = torch.flatten(x, start_dim=1)
        features = x
        for i in range(len(self.linear_layers)):
            x = self.tanh(self.batch_norm_layers[i](self.linear_layers[i](x)))

        x = self.softmax(x)
        return x, features

    def init_weights(self):
        for layer in self.conv_layers:
            kaiming_uniform_(layer.weight)
            normal_(layer.bias)

        for layer in self.linear_layers:
            xavier_uniform_(layer.weight)
            normal_(layer.bias)