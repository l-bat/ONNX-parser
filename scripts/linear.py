import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np

model = nn.Linear(3, 4, bias=True)


input = autograd.Variable(torch.randn(2, 3))
print(input.shape)
np.save("data/input_linear", input)

output = model(input)
print(output.shape)
np.save("data/output_linear", output.data)


torch.onnx.export(model, input, "model/linear.onnx",  export_params=True)
