import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from termcolor import colored


def save_data_and_model(name, input, model):
    print name + " input has sizes" , colored(input.shape, 'green')
    np.save("input_" + name, input.data)
    output = model(input)
    print name + " output has sizes" , colored(output.shape, 'red')
    np.save("output_" + name, output.data)
    torch.onnx.export(model, input, name + ".onnx", export_params=True)

torch.manual_seed(7)

input = Variable(torch.randn(20, 3, 50, 100))
max_pool = nn.MaxPool2d(kernel_size=(5,3), stride=1, padding=0, dilation=1)
save_data_and_model("maxpool", input, max_pool)


input = Variable(torch.randn(20, 3, 50, 100))
conv = nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=0)
save_data_and_model("convolution", input, conv)


input = Variable(torch.randn(2, 3))
linear = nn.Linear(3, 4, bias=True)
save_data_and_model("linear", input, linear)

model = nn.Sequential(
          nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
          nn.Sigmoid()
        )
input = autograd.Variable(torch.randn(4, 3, 12, 18))
save_data_and_model("maxpool_sigmoid", input, model)


input = Variable(torch.randn(20, 3, 50, 100))
conv2 = nn.Sequential(
          nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=0),
          nn.Conv2d(16, 24, kernel_size=5, stride=1, padding=0)
          )
save_data_and_model("conv2", input, conv2)


input = Variable(torch.randn(4, 3, 30, 45))
maxpool2 = nn.Sequential(
           nn.MaxPool2d(kernel_size=5, stride=5, padding=0, dilation=1),
           nn.MaxPool2d(kernel_size=3, stride=3, padding=0, dilation=1)
           )
save_data_and_model("maxpool2", input, maxpool2)


input = Variable(torch.randn(1, 64, 55, 55))
relu = nn.ReLU(inplace=True)
save_data_and_model("ReLU", input, relu)


input = Variable(torch.randn(2, 3))
dropout = nn.Dropout()
dropout.eval()
save_data_and_model("dropout", input, dropout)
