import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from termcolor import colored


torch.manual_seed(7)

input = autograd.Variable(torch.randn(20, 3, 50, 100))
print "Generate input for maxpooling with sizes",  colored(colored(input.shape, 'green'), 'green')
max_pool = nn.MaxPool2d(kernel_size=(5,3), stride=1, padding=0, dilation=1)
output = max_pool(input)
print"Maxpooling output has sizes" , colored(output.shape, 'red')
np.save("data/input_maxpool", input)
np.save("data/output_maxpool", output.data)
torch.onnx.export(max_pool, input, "model/maxpooling.onnx", export_params=True)


conv = nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=0)
input = autograd.Variable(torch.randn(20, 3, 50, 100))
print "Generate input for convolution with sizes" , colored(input.shape, 'green')
np.save("data/input_conv", input)
output = conv(input)
print "Convolution output has sizes" , colored(output.shape, 'red')
np.save("data/output_conv", output.data)
torch.onnx.export(conv, input, "model/convolution.onnx", export_params=True)


linear = nn.Linear(3, 4, bias=True)
input = autograd.Variable(torch.randn(2, 3))
print "Generate input for linear with sizes" , colored(input.shape, 'green')
np.save("data/input_linear", input)
output = linear(input)
print "Generate output for linear with sizes" , colored(output.shape, 'red')
np.save("data/output_linear", output.data)
torch.onnx.export(linear, input, "model/linear.onnx",  export_params=True)


model = nn.Sequential(
          nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
          nn.Sigmoid()
        )
input = autograd.Variable(torch.randn(4, 3, 12, 18))
print "Generate input for maxpool + sigmoid with sizes" , colored(input.shape, 'green')
output = model(input)
np.save("data/input_LRN", input)
print "Generate output for maxpool + sigmoid with sizes" , colored(output.shape, 'red')
np.save("data/output_LRN", output.data)
torch.onnx.export(model, input, "model/max_pool_LRN.onnx",  export_params=True)


conv2 = nn.Sequential(
          nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=0),
          nn.Conv2d(16, 24, kernel_size=5, stride=1, padding=0)
          )
input = autograd.Variable(torch.randn(20, 3, 50, 100))
print "Generate input for conv + conv with sizes", colored(input.shape, 'green')
np.save("data/input_2conv", input)
output = conv2(input)
print "Generate output for conv + conv with sizes", colored(output.shape, 'red')
np.save("data/output_2conv", output.data)
torch.onnx.export(conv2, input, "model/conv2.onnx",  export_params=True)


maxpool2 = nn.Sequential(
           nn.MaxPool2d(kernel_size=5, stride=5, padding=0, dilation=1),
           nn.MaxPool2d(kernel_size=3, stride=3, padding=0, dilation=1)
           )
input = autograd.Variable(torch.randn(4, 3, 30, 45))
output = maxpool2(input)
print "Generate input for maxpool + maxpool with sizes", colored(input.shape, 'green')
np.save("data/input_2max_pool", input)
print "Generate output for maxpool + maxpool with sizes", colored(output.shape, 'red')
np.save("data/output_2max_pool", output)
torch.onnx.export(maxpool2, input, "model/two_maxpool.onnx",  export_params=True)
