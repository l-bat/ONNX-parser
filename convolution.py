import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torchvision.transforms as transforms



# batch_size = 1
# stride = 1
# kernel_size = 5
# padding = 0
# how to compute out_channels (int): Number of channels produced by the convolution


# class ConvNet(nn.Module):
#     def __init__(self):  # def __init__(self, num_classes=10):
#         super(ConvNet, self).__init__()
#         self.layer1 = nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2)
#
#
#     def forward(self, x):
#         x = self.layer1(x)
#         return x
#
# def convolution(**kwargs):
#     model = ConvNet(**kwargs)
#     return model


conv = nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=0)
input = autograd.Variable(torch.randn(20, 3, 50, 100))
output = conv(input)
# output = convolution(input)
print(input.shape)
print(output.shape)

torch.onnx.export(conv, input, "convolution.onnx")
