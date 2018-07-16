import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np

#max_pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=0, dilation=1)
model = nn.Sequential(
          nn.MaxPool2d(kernel_size=5, stride=5, padding=0, dilation=1),
          nn.MaxPool2d(kernel_size=3, stride=3, padding=0, dilation=1)
        )
input = autograd.Variable(torch.randn(4, 3, 30, 45))
output = model(input)

print(input.shape)
np.save("data/input_2max_pool", input)
print(output.shape)
np.save("data/output_2max_pool", output)

torch.onnx.export(model, input, "model/two_maxpool.onnx")
