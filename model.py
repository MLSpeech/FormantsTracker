import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import LambdaLayer


#======================2D===================#
# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1,dilation=1,bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1,dilation=dilation, bias=bias)


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,dilation=1, downsample=None,bias=True):
        super(ResidualBlock, self).__init__()
        self.out_channels=out_channels
        self.conv1 = conv3x3(in_channels, out_channels, stride,dilation=dilation,bias=bias)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels,stride,dilation=dilation,bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


def make_layer(block, in_channels, out_channels, blocks, stride=1,drop=0,dilation=1,bias=True):
    downsample = None
    if (stride != 1) or (in_channels != out_channels):
        downsample = nn.Sequential(
            conv3x3(in_channels, out_channels, stride=stride,bias=bias),
            nn.BatchNorm2d(out_channels))
    layers = []
    layers.append(block(in_channels, out_channels, stride, dilation,downsample,bias=bias))
    layers.append(nn.Dropout(p=drop))
    in_channels = out_channels
    for i in range(1, blocks):
        layers.append(block(out_channels, out_channels,bias=bias))
        layers.append(nn.Dropout(p=drop))
    return nn.Sequential(*layers)


#======================1D===================#
# Residual block
class ResidualBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None,bias=True):
        super(ResidualBlock1d, self).__init__()
        self.out_channels=out_channels
        self.conv1 = conv1d(in_channels, out_channels, stride,bias=bias)
        self.bn1 = nn.BatchNorm1d(out_channels,affine=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1d(out_channels, out_channels,stride,bias=bias)
        self.bn2 = nn.BatchNorm1d(out_channels,affine=False)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

def make_layer_1d(block, in_channels, out_channels, blocks, stride,drop,bias):
    downsample = None
    if (stride != 1) or (in_channels != out_channels):
        downsample = nn.Sequential(
            conv1d(in_channels, out_channels, stride=stride,bias=bias),
            nn.BatchNorm1d(out_channels))
    layers = []
    layers.append(block(in_channels, out_channels, stride, downsample,bias=bias))
    in_channels = out_channels
    for i in range(1, blocks):
        layers.append(nn.Dropout(p=drop))
        layers.append(block(out_channels, out_channels,bias=bias))
    return nn.Sequential(*layers)

def conv1d(in_channels, out_channels, stride=1,bias=True,):
    return nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1,
                     padding=1, bias=bias) 




class FormantTracker(nn.Module):
    def __init__(self, hparams):
        super(FormantTracker, self).__init__()
        self.hparams = hparams
        self.n_bins = (self.hparams.n_fft//2)+1

        self.enc = nn.Sequential(
            LambdaLayer(lambda x: x.unsqueeze(1)),
            make_layer(ResidualBlock, in_channels=1, out_channels=16, blocks=1, stride=1,drop=self.hparams.dropout,bias=self.hparams.bias),
            make_layer(ResidualBlock, in_channels=16, out_channels=32, blocks=1, stride=1,drop=self.hparams.dropout,bias=self.hparams.bias),
            make_layer(ResidualBlock, in_channels=32, out_channels=64, blocks=1, stride=1,drop=self.hparams.dropout,bias=self.hparams.bias),
            make_layer(ResidualBlock, in_channels=64, out_channels=128, blocks=1, stride=1,drop=self.hparams.dropout,bias=self.hparams.bias),
            make_layer(ResidualBlock, in_channels=128, out_channels=128, blocks=4, stride=1,drop=self.hparams.dropout,bias=self.hparams.bias),
            make_layer(ResidualBlock, in_channels=128, out_channels=64, blocks=1, stride=1,drop=self.hparams.dropout,bias=self.hparams.bias),
            make_layer(ResidualBlock, in_channels=64, out_channels=32, blocks=1, stride=1,drop=self.hparams.dropout,bias=self.hparams.bias),
            make_layer(ResidualBlock, in_channels=32, out_channels=16, blocks=1, stride=1,drop=self.hparams.dropout,bias=self.hparams.bias),
            make_layer(ResidualBlock, in_channels=16, out_channels=1, blocks=1, stride=1,drop=self.hparams.dropout,bias=self.hparams.bias),
            LambdaLayer(lambda x: x.squeeze(1)),
            LambdaLayer(lambda x: x.transpose(1,2))
        )
            
        self.f1_conv=nn.Sequential(make_layer_1d(ResidualBlock1d, in_channels=self.n_bins, out_channels=self.n_bins//4, blocks=self.hparams.f1_blocks, stride=1,drop=self.hparams.dropout,bias=self.hparams.bias1d),
                        make_layer_1d(ResidualBlock1d, in_channels=self.n_bins//4, out_channels=self.n_bins, blocks=self.hparams.f1_blocks, stride=1,drop=self.hparams.dropout,bias=self.hparams.bias1d))
        self.f2_conv=nn.Sequential(make_layer_1d(ResidualBlock1d, in_channels=self.n_bins, out_channels=self.n_bins//4, blocks=self.hparams.f2_blocks, stride=1,drop=self.hparams.dropout,bias=self.hparams.bias1d),
                        make_layer_1d(ResidualBlock1d, in_channels=self.n_bins//4, out_channels=self.n_bins, blocks=self.hparams.f2_blocks, stride=1,drop=self.hparams.dropout,bias=self.hparams.bias1d))
        self.f3_conv=nn.Sequential(make_layer_1d(ResidualBlock1d, in_channels=self.n_bins, out_channels=self.n_bins//4, blocks=self.hparams.f3_blocks, stride=1,drop=self.hparams.dropout,bias=self.hparams.bias1d),
                        make_layer_1d(ResidualBlock1d, in_channels=self.n_bins//4, out_channels=self.n_bins, blocks=self.hparams.f3_blocks, stride=1,drop=self.hparams.dropout,bias=self.hparams.bias1d))



    def forward(self, spect):
        h = self.enc(spect)
        out1 = self.f1_conv(h)
        out2 = self.f2_conv(h-out1)
        out3 = self.f3_conv(h-out1-out2)
        out= torch.stack([out1,out2,out3],dim=1)
        out=out.transpose(2,3)
        return out ,h
    
