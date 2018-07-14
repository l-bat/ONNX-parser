import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np

help(nn)

features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Linear(256 * 6 * 6, 4096),
        )

classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 20),  #nn.Linear(4096, num_classes=1),
        )

def alexnet(input):
    output = features(input)
    output = output.view(output.size(0), 256 * 6 * 6)
    output = classifier(output)
    return output


#input = autograd.Variable(torch.randn(4, 3, 240, 240))
input = autograd.Variable(torch.randn(1, 3, 227, 227))
print(input.shape)
np.save("data/input_alexnet", input)
output = features(input)
print(output.shape)
#output = output.view(output.size(0), 256 * 6 * 6)
#output = classifier(output)

#output = alexnet(input)

print(output.shape)
np.save("data/output_alexnet", output.data)

torch.onnx.export(features, input, "model/alexnet.onnx")
