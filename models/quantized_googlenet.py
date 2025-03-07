import torch
import torch.nn as nn
from torch.ao.nn.quantized import FloatFunctional

class Inception(nn.Module):
    def __init__(self, input_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj):
        super().__init__()

        #1x1conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(input_channels, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(inplace=True)
        )

        #1x1conv -> 3x3conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(input_channels, n3x3_reduce, kernel_size=1),
            nn.BatchNorm2d(n3x3_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(n3x3_reduce, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(inplace=True)
        )

        #1x1conv -> 5x5conv branch
        #we use 2 3x3 conv filters stacked instead
        #of 1 5x5 filters to obtain the same receptive
        #field with fewer parameters
        self.b3 = nn.Sequential(
            nn.Conv2d(input_channels, n5x5_reduce, kernel_size=1),
            nn.BatchNorm2d(n5x5_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5_reduce, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5, n5x5),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(inplace=True)
        )

        #3x3pooling -> 1x1conv
        #same conv
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(input_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        ff = FloatFunctional()
        return ff.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


class Quantized_Googlenet(nn.Module):

    def __init__(self, num_class=100):
        super().__init__()
        self.prelayer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )

        #although we only use 1 conv layer as prelayer,
        #we still use name a3, b3.......
        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        ##"""In general, an Inception network is a network consisting of
        ##modules of the above type stacked upon each other, with occasional
        ##max-pooling layers with stride 2 to halve the resolution of the
        ##grid"""
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        #input feature size: 8*8*1024
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d(p=0.4)
        self.linear = nn.Linear(1024, num_class)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
    
    def forward(self, x):
        x = torch.quantize_per_tensor(x, 0.1, 10, torch.quint8)
        x = self.prelayer(x)
        x = self.maxpool(x)
        x = self.a3(x)
        x = self.b3(x)

        x = self.maxpool(x)

        x = self.a4(x)
        x = self.b4(x)
        x = self.c4(x)
        x = self.d4(x)
        x = self.e4(x)

        x = self.maxpool(x)

        x = self.a5(x)
        x = self.b5(x)

        #"""It was found that a move from fully connected layers to
        #average pooling improved the top-1 accuracy by about 0.6%,
        #however the use of dropout remained essential even after
        #removing the fully connected layers."""
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(x.size()[0], -1)
        x = torch.dequantize(x)

        x = self.linear(x)
        

        return x
    
def quantized_googlenet():
    quantized_model = Quantized_Googlenet()

    quantized_model.eval()

    quantized_model.prelayer.qconfig = torch.quantization.get_default_qconfig('x86')

    
    modules_to_fuse = [
    ['prelayer.0', 'prelayer.1', 'prelayer.2'],
    ['prelayer.3', 'prelayer.4', 'prelayer.5'],
    ['prelayer.6', 'prelayer.7', 'prelayer.8'],

    ['a3.b1.0', 'a3.b1.1', 'a3.b1.2'],
    ['a3.b2.0', 'a3.b2.1', 'a3.b2.2'],
    ['a3.b2.3', 'a3.b2.4', 'a3.b2.5'],
    ['a3.b3.0', 'a3.b3.1', 'a3.b3.2'],
    ['a3.b3.3', 'a3.b3.4', 'a3.b3.5'],
    ['a3.b3.6', 'a3.b3.7', 'a3.b3.8'],
    ['a3.b4.1', 'a3.b4.2', 'a3.b4.3'],

    ['b3.b1.0', 'b3.b1.1', 'b3.b1.2'],
    ['b3.b2.0', 'b3.b2.1', 'b3.b2.2'],
    ['b3.b2.3', 'b3.b2.4', 'b3.b2.5'],
    ['b3.b3.0', 'b3.b3.1', 'b3.b3.2'],
    ['b3.b3.3', 'b3.b3.4', 'b3.b3.5'],
    ['b3.b3.6', 'b3.b3.7', 'b3.b3.8'],
    ['b3.b4.1', 'b3.b4.2', 'b3.b4.3'],

    ['a4.b1.0', 'a4.b1.1', 'a4.b1.2'],
    ['a4.b2.0', 'a4.b2.1', 'a4.b2.2'],
    ['a4.b2.3', 'a4.b2.4', 'a4.b2.5'],
    ['a4.b3.0', 'a4.b3.1', 'a4.b3.2'],
    ['a4.b3.3', 'a4.b3.4', 'a4.b3.5'],
    ['a4.b3.6', 'a4.b3.7', 'a4.b3.8'],
    ['a4.b4.1', 'a4.b4.2', 'a4.b4.3'],

    ['b4.b1.0', 'b4.b1.1', 'b4.b1.2'],
    ['b4.b2.0', 'b4.b2.1', 'b4.b2.2'],
    ['b4.b2.3', 'b4.b2.4', 'b4.b2.5'],
    ['b4.b3.0', 'b4.b3.1', 'b4.b3.2'],
    ['b4.b3.3', 'b4.b3.4', 'b4.b3.5'],
    ['b4.b3.6', 'b4.b3.7', 'b4.b3.8'],
    ['b4.b4.1', 'b4.b4.2', 'b4.b4.3'],

    ['c4.b1.0', 'c4.b1.1', 'c4.b1.2'],
    ['c4.b2.0', 'c4.b2.1', 'c4.b2.2'],
    ['c4.b2.3', 'c4.b2.4', 'c4.b2.5'],
    ['c4.b3.0', 'c4.b3.1', 'c4.b3.2'],
    ['c4.b3.3', 'c4.b3.4', 'c4.b3.5'],
    ['c4.b3.6', 'c4.b3.7', 'c4.b3.8'],
    ['c4.b4.1', 'c4.b4.2', 'c4.b4.3'],

    ['d4.b1.0', 'd4.b1.1', 'd4.b1.2'],
    ['d4.b2.0', 'd4.b2.1', 'd4.b2.2'],
    ['d4.b2.3', 'd4.b2.4', 'd4.b2.5'],
    ['d4.b3.0', 'd4.b3.1', 'd4.b3.2'],
    ['d4.b3.3', 'd4.b3.4', 'd4.b3.5'],
    ['d4.b3.6', 'd4.b3.7', 'd4.b3.8'],
    ['d4.b4.1', 'd4.b4.2', 'd4.b4.3'],

    ['e4.b1.0', 'e4.b1.1', 'e4.b1.2'],
    ['e4.b2.0', 'e4.b2.1', 'e4.b2.2'],
    ['e4.b2.3', 'e4.b2.4', 'e4.b2.5'],
    ['e4.b3.0', 'e4.b3.1', 'e4.b3.2'],
    ['e4.b3.3', 'e4.b3.4', 'e4.b3.5'],
    ['e4.b3.6', 'e4.b3.7', 'e4.b3.8'],
    ['e4.b4.1', 'e4.b4.2', 'e4.b4.3'],

    ['a5.b1.0', 'a5.b1.1', 'a5.b1.2'],
    ['a5.b2.0', 'a5.b2.1', 'a5.b2.2'],
    ['a5.b2.3', 'a5.b2.4', 'a5.b2.5'],
    ['a5.b3.0', 'a5.b3.1', 'a5.b3.2'],
    ['a5.b3.3', 'a5.b3.4', 'a5.b3.5'],
    ['a5.b3.6', 'a5.b3.7', 'a5.b3.8'],
    ['a5.b4.1', 'a5.b4.2', 'a5.b4.3'],

    ['b5.b1.0', 'b5.b1.1', 'b5.b1.2'],
    ['b5.b2.0', 'b5.b2.1', 'b5.b2.2'],
    ['b5.b2.3', 'b5.b2.4', 'b5.b2.5'],
    ['b5.b3.0', 'b5.b3.1', 'b5.b3.2'],
    ['b5.b3.3', 'b5.b3.4', 'b5.b3.5'],
    ['b5.b3.6', 'b5.b3.7', 'b5.b3.8'],
    ['b5.b4.1', 'b5.b4.2', 'b5.b4.3']
    ]

    quantized_model_fused = torch.quantization.fuse_modules(quantized_model, modules_to_fuse)

    quantized_model = torch.quantization.prepare(quantized_model_fused)

    model_int8 = torch.quantization.convert(quantized_model)

    return model_int8, quantized_model