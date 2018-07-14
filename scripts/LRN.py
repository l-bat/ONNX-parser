import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np

#model = nn.LocalResponseNorm(size=3)
model = nn.Sequential(
          nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
         # nn.LocalResponseNorm(size=3)
          nn.Sigmoid()
        )
input = autograd.Variable(torch.randn(4, 3, 12, 18))
output = model(input)
print(output)
print(input.shape)
np.save("data/input_LRN", input)
print(output.shape)
np.save("data/output_LRN", output.data)

torch.onnx.export(model, input, "model/max_pool_LRN.onnx")
