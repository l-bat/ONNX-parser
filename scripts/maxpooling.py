import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np

max_pool = nn.MaxPool2d((5,3), stride=1, padding=0, dilation=1)
input = autograd.Variable(torch.randn(20, 3, 50, 100))
output = max_pool(input)

#print(input.shape)
np.save("data/input_maxpool", input)
#np.load("input.npy")
#print(output.shape)
np.save("data/output_maxpool", output)
#np.load("output.npy")

torch.onnx.export(max_pool, input, "model/maxpooling.onnx")
