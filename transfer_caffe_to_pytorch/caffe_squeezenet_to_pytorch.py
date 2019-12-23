import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__weights_dict = dict()

def load_weights(weight_file):
    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file, allow_pickle=True).item()
    except:
        weights_dict = np.load(weight_file, allow_pickle=True, encoding='bytes').item()

    return weights_dict

class KitModel(nn.Module):

    
    def __init__(self, weight_file):
        super(KitModel, self).__init__()
        global __weights_dict
        __weights_dict = load_weights(weight_file)

        self.conv1 = self.__conv(2, name='conv1', in_channels=3, out_channels=64, kernel_size=(3L, 3L), stride=(2L, 2L), groups=1, bias=True)
        self.fire2_squeeze1x1 = self.__conv(2, name='fire2/squeeze1x1', in_channels=64, out_channels=16, kernel_size=(1L, 1L), stride=(1L, 1L), groups=1, bias=True)
        self.fire2_expand1x1 = self.__conv(2, name='fire2/expand1x1', in_channels=16, out_channels=64, kernel_size=(1L, 1L), stride=(1L, 1L), groups=1, bias=True)
        self.fire2_expand3x3 = self.__conv(2, name='fire2/expand3x3', in_channels=16, out_channels=64, kernel_size=(3L, 3L), stride=(1L, 1L), groups=1, bias=True)
        self.fire3_squeeze1x1 = self.__conv(2, name='fire3/squeeze1x1', in_channels=128, out_channels=16, kernel_size=(1L, 1L), stride=(1L, 1L), groups=1, bias=True)
        self.fire3_expand3x3 = self.__conv(2, name='fire3/expand3x3', in_channels=16, out_channels=64, kernel_size=(3L, 3L), stride=(1L, 1L), groups=1, bias=True)
        self.fire3_expand1x1 = self.__conv(2, name='fire3/expand1x1', in_channels=16, out_channels=64, kernel_size=(1L, 1L), stride=(1L, 1L), groups=1, bias=True)
        self.fire4_squeeze1x1 = self.__conv(2, name='fire4/squeeze1x1', in_channels=128, out_channels=32, kernel_size=(1L, 1L), stride=(1L, 1L), groups=1, bias=True)
        self.fire4_expand1x1 = self.__conv(2, name='fire4/expand1x1', in_channels=32, out_channels=128, kernel_size=(1L, 1L), stride=(1L, 1L), groups=1, bias=True)
        self.fire4_expand3x3 = self.__conv(2, name='fire4/expand3x3', in_channels=32, out_channels=128, kernel_size=(3L, 3L), stride=(1L, 1L), groups=1, bias=True)
        self.fire5_squeeze1x1 = self.__conv(2, name='fire5/squeeze1x1', in_channels=256, out_channels=32, kernel_size=(1L, 1L), stride=(1L, 1L), groups=1, bias=True)
        self.fire5_expand3x3 = self.__conv(2, name='fire5/expand3x3', in_channels=32, out_channels=128, kernel_size=(3L, 3L), stride=(1L, 1L), groups=1, bias=True)
        self.fire5_expand1x1 = self.__conv(2, name='fire5/expand1x1', in_channels=32, out_channels=128, kernel_size=(1L, 1L), stride=(1L, 1L), groups=1, bias=True)
        self.fire6_squeeze1x1 = self.__conv(2, name='fire6/squeeze1x1', in_channels=256, out_channels=48, kernel_size=(1L, 1L), stride=(1L, 1L), groups=1, bias=True)
        self.fire6_expand1x1 = self.__conv(2, name='fire6/expand1x1', in_channels=48, out_channels=192, kernel_size=(1L, 1L), stride=(1L, 1L), groups=1, bias=True)
        self.fire6_expand3x3 = self.__conv(2, name='fire6/expand3x3', in_channels=48, out_channels=192, kernel_size=(3L, 3L), stride=(1L, 1L), groups=1, bias=True)
        self.fire7_squeeze1x1 = self.__conv(2, name='fire7/squeeze1x1', in_channels=384, out_channels=48, kernel_size=(1L, 1L), stride=(1L, 1L), groups=1, bias=True)
        self.fire7_expand1x1 = self.__conv(2, name='fire7/expand1x1', in_channels=48, out_channels=192, kernel_size=(1L, 1L), stride=(1L, 1L), groups=1, bias=True)
        self.fire7_expand3x3 = self.__conv(2, name='fire7/expand3x3', in_channels=48, out_channels=192, kernel_size=(3L, 3L), stride=(1L, 1L), groups=1, bias=True)
        self.fire8_squeeze1x1 = self.__conv(2, name='fire8/squeeze1x1', in_channels=384, out_channels=64, kernel_size=(1L, 1L), stride=(1L, 1L), groups=1, bias=True)
        self.fire8_expand1x1 = self.__conv(2, name='fire8/expand1x1', in_channels=64, out_channels=256, kernel_size=(1L, 1L), stride=(1L, 1L), groups=1, bias=True)
        self.fire8_expand3x3 = self.__conv(2, name='fire8/expand3x3', in_channels=64, out_channels=256, kernel_size=(3L, 3L), stride=(1L, 1L), groups=1, bias=True)
        self.fire9_squeeze1x1 = self.__conv(2, name='fire9/squeeze1x1', in_channels=512, out_channels=64, kernel_size=(1L, 1L), stride=(1L, 1L), groups=1, bias=True)
        self.fire9_expand1x1 = self.__conv(2, name='fire9/expand1x1', in_channels=64, out_channels=256, kernel_size=(1L, 1L), stride=(1L, 1L), groups=1, bias=True)
        self.fire9_expand3x3 = self.__conv(2, name='fire9/expand3x3', in_channels=64, out_channels=256, kernel_size=(3L, 3L), stride=(1L, 1L), groups=1, bias=True)
        self.my_conv10_tesla = self.__conv(2, name='my-conv10-tesla', in_channels=512, out_channels=9, kernel_size=(1L, 1L), stride=(1L, 1L), groups=1, bias=True)

    def forward(self, x):
        conv1_pad       = F.pad(x, (0L, 1L, 0L, 1L))
        conv1           = self.conv1(conv1_pad)
        relu_conv1      = F.relu(conv1)
        pool1_pad       = F.pad(relu_conv1, (0L, 1L, 0L, 1L), value=float('-inf'))
        pool1           = F.max_pool2d(pool1_pad, kernel_size=(3L, 3L), stride=(2L, 2L), padding=0, ceil_mode=False)
        fire2_squeeze1x1 = self.fire2_squeeze1x1(pool1)
        fire2_relu_squeeze1x1 = F.relu(fire2_squeeze1x1)
        fire2_expand1x1 = self.fire2_expand1x1(fire2_relu_squeeze1x1)
        fire2_expand3x3_pad = F.pad(fire2_relu_squeeze1x1, (1L, 1L, 1L, 1L))
        fire2_expand3x3 = self.fire2_expand3x3(fire2_expand3x3_pad)
        fire2_relu_expand1x1 = F.relu(fire2_expand1x1)
        fire2_relu_expand3x3 = F.relu(fire2_expand3x3)
        fire2_concat    = torch.cat((fire2_relu_expand1x1, fire2_relu_expand3x3), 1)
        fire3_squeeze1x1 = self.fire3_squeeze1x1(fire2_concat)
        fire3_relu_squeeze1x1 = F.relu(fire3_squeeze1x1)
        fire3_expand3x3_pad = F.pad(fire3_relu_squeeze1x1, (1L, 1L, 1L, 1L))
        fire3_expand3x3 = self.fire3_expand3x3(fire3_expand3x3_pad)
        fire3_expand1x1 = self.fire3_expand1x1(fire3_relu_squeeze1x1)
        fire3_relu_expand3x3 = F.relu(fire3_expand3x3)
        fire3_relu_expand1x1 = F.relu(fire3_expand1x1)
        fire3_concat    = torch.cat((fire3_relu_expand1x1, fire3_relu_expand3x3), 1)
        pool3_pad       = F.pad(fire3_concat, (0L, 1L, 0L, 1L), value=float('-inf'))
        pool3           = F.max_pool2d(pool3_pad, kernel_size=(3L, 3L), stride=(2L, 2L), padding=0, ceil_mode=False)
        fire4_squeeze1x1 = self.fire4_squeeze1x1(pool3)
        fire4_relu_squeeze1x1 = F.relu(fire4_squeeze1x1)
        fire4_expand1x1 = self.fire4_expand1x1(fire4_relu_squeeze1x1)
        fire4_expand3x3_pad = F.pad(fire4_relu_squeeze1x1, (1L, 1L, 1L, 1L))
        fire4_expand3x3 = self.fire4_expand3x3(fire4_expand3x3_pad)
        fire4_relu_expand1x1 = F.relu(fire4_expand1x1)
        fire4_relu_expand3x3 = F.relu(fire4_expand3x3)
        fire4_concat    = torch.cat((fire4_relu_expand1x1, fire4_relu_expand3x3), 1)
        fire5_squeeze1x1 = self.fire5_squeeze1x1(fire4_concat)
        fire5_relu_squeeze1x1 = F.relu(fire5_squeeze1x1)
        fire5_expand3x3_pad = F.pad(fire5_relu_squeeze1x1, (1L, 1L, 1L, 1L))
        fire5_expand3x3 = self.fire5_expand3x3(fire5_expand3x3_pad)
        fire5_expand1x1 = self.fire5_expand1x1(fire5_relu_squeeze1x1)
        fire5_relu_expand3x3 = F.relu(fire5_expand3x3)
        fire5_relu_expand1x1 = F.relu(fire5_expand1x1)
        fire5_concat    = torch.cat((fire5_relu_expand1x1, fire5_relu_expand3x3), 1)
        pool5_pad       = F.pad(fire5_concat, (0L, 1L, 0L, 1L), value=float('-inf'))
        pool5           = F.max_pool2d(pool5_pad, kernel_size=(3L, 3L), stride=(2L, 2L), padding=0, ceil_mode=False)
        fire6_squeeze1x1 = self.fire6_squeeze1x1(pool5)
        fire6_relu_squeeze1x1 = F.relu(fire6_squeeze1x1)
        fire6_expand1x1 = self.fire6_expand1x1(fire6_relu_squeeze1x1)
        fire6_expand3x3_pad = F.pad(fire6_relu_squeeze1x1, (1L, 1L, 1L, 1L))
        fire6_expand3x3 = self.fire6_expand3x3(fire6_expand3x3_pad)
        fire6_relu_expand1x1 = F.relu(fire6_expand1x1)
        fire6_relu_expand3x3 = F.relu(fire6_expand3x3)
        fire6_concat    = torch.cat((fire6_relu_expand1x1, fire6_relu_expand3x3), 1)
        fire7_squeeze1x1 = self.fire7_squeeze1x1(fire6_concat)
        fire7_relu_squeeze1x1 = F.relu(fire7_squeeze1x1)
        fire7_expand1x1 = self.fire7_expand1x1(fire7_relu_squeeze1x1)
        fire7_expand3x3_pad = F.pad(fire7_relu_squeeze1x1, (1L, 1L, 1L, 1L))
        fire7_expand3x3 = self.fire7_expand3x3(fire7_expand3x3_pad)
        fire7_relu_expand1x1 = F.relu(fire7_expand1x1)
        fire7_relu_expand3x3 = F.relu(fire7_expand3x3)
        fire7_concat    = torch.cat((fire7_relu_expand1x1, fire7_relu_expand3x3), 1)
        fire8_squeeze1x1 = self.fire8_squeeze1x1(fire7_concat)
        fire8_relu_squeeze1x1 = F.relu(fire8_squeeze1x1)
        fire8_expand1x1 = self.fire8_expand1x1(fire8_relu_squeeze1x1)
        fire8_expand3x3_pad = F.pad(fire8_relu_squeeze1x1, (1L, 1L, 1L, 1L))
        fire8_expand3x3 = self.fire8_expand3x3(fire8_expand3x3_pad)
        fire8_relu_expand1x1 = F.relu(fire8_expand1x1)
        fire8_relu_expand3x3 = F.relu(fire8_expand3x3)
        fire8_concat    = torch.cat((fire8_relu_expand1x1, fire8_relu_expand3x3), 1)
        fire9_squeeze1x1 = self.fire9_squeeze1x1(fire8_concat)
        fire9_relu_squeeze1x1 = F.relu(fire9_squeeze1x1)
        fire9_expand1x1 = self.fire9_expand1x1(fire9_relu_squeeze1x1)
        fire9_expand3x3_pad = F.pad(fire9_relu_squeeze1x1, (1L, 1L, 1L, 1L))
        fire9_expand3x3 = self.fire9_expand3x3(fire9_expand3x3_pad)
        fire9_relu_expand1x1 = F.relu(fire9_expand1x1)
        fire9_relu_expand3x3 = F.relu(fire9_expand3x3)
        fire9_concat    = torch.cat((fire9_relu_expand1x1, fire9_relu_expand3x3), 1)
        drop9           = F.dropout(input = fire9_concat, p = 0.5, training = self.training, inplace = True)
        my_conv10_tesla = self.my_conv10_tesla(drop9)
        relu_conv10     = F.relu(my_conv10_tesla)
        heatmap         = torch.cat((relu_conv10), 1)
        pool10          = F.avg_pool2d(relu_conv10, kernel_size=(14L, 14L), stride=(1L, 1L), padding=(0L,), ceil_mode=False, count_include_pad=False)
        prob            = F.softmax(pool10)
        return heatmap, prob


    @staticmethod
    def __conv(dim, name, **kwargs):
        if   dim == 1:  layer = nn.Conv1d(**kwargs)
        elif dim == 2:  layer = nn.Conv2d(**kwargs)
        elif dim == 3:  layer = nn.Conv3d(**kwargs)
        else:           raise NotImplementedError()

        layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['weights']))
        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        return layer

