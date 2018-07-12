import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torchvision.transforms as transforms


max_pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=0, dilation=1)
input = autograd.Variable(torch.randn(20, 3, 50, 100))
output = max_pool(input)
# output = convolution(input)
print(input.shape)
print(output.shape)


torch.onnx.export(max_pool, input, "maxpooling.onnx")
