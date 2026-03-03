import ltn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from torch.nn.init import xavier_uniform_, normal_, kaiming_uniform_

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

        self.init_weights()

    def forward(self, x):
        for conv in self.conv_layers:
            x = self.relu(conv(x))
            x = self.maxpool(x)
        x = torch.flatten(x, start_dim=1)
        for i in range(len(self.linear_layers)):
            x = self.tanh(self.batch_norm_layers[i](self.linear_layers[i](x)))
        return x

    def init_weights(self):
        for layer in self.conv_layers:
            kaiming_uniform_(layer.weight)
            normal_(layer.bias)

        for layer in self.linear_layers:
            xavier_uniform_(layer.weight)
            normal_(layer.bias)


class SingleDigitClassifier(torch.nn.Module):
    def __init__(self, layers_sizes=(100, 84, 10)):
        super(SingleDigitClassifier, self).__init__()
        self.mnistconv = MNISTConv()  # this is the convolutional part of the architecture
        self.tanh = torch.nn.Tanh()  # tanh is used as activation for the linear layers
        self.linear_layers = torch.nn.ModuleList([torch.nn.Linear(layers_sizes[i - 1], layers_sizes[i])
                                                  for i in range(1, len(layers_sizes))])
        self.batch_norm_layers = torch.nn.ModuleList([torch.nn.BatchNorm1d(layers_sizes[i])
                                                      for i in range(1, len(layers_sizes))])
        self.init_weights()

    def forward(self, x):
        x = self.mnistconv(x)
        for i in range(len(self.linear_layers) - 1):
            x = self.tanh(self.batch_norm_layers[i](self.linear_layers[i](x)))
        return self.linear_layers[-1](x)  # in the last layer a sigmoid or a softmax has to be applied

    def init_weights(self):
        for layer in self.linear_layers:
            xavier_uniform_(layer.weight)
            normal_(layer.bias)


class LogitsToProbability(torch.nn.Module):
    def __init__(self, logits_model):
        super(LogitsToProbability, self).__init__()
        self.logits_model = logits_model
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, d):
        logits = self.logits_model(x)
        probs = self.softmax(logits)
        out = torch.gather(probs, 1, d)
        return out