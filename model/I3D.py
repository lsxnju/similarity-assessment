import torch
import torch.nn.functional as F
import os

# 计算padding
def get_padding_shape(filter_shape, stride):
    def _pad_top_bottom(filter_dim, stride_val):
        pad_along = max(filter_dim - stride_val, 0)
        pad_top = pad_along // 2
        pad_bottom = pad_along - pad_top
        return pad_top, pad_bottom

    padding_shape = []
    for filter_dim, stride_val in zip(filter_shape, stride):
        pad_top, pad_bottom = _pad_top_bottom(filter_dim, stride_val)
        padding_shape.append(pad_top)
        padding_shape.append(pad_bottom)
    depth_top = padding_shape.pop(0)
    depth_bottom = padding_shape.pop(0)
    padding_shape.append(depth_top)
    padding_shape.append(depth_bottom)
    return tuple(padding_shape)

def simplify_padding(padding_shapes):
    all_same = True
    padding_init = padding_shapes[0]
    for pad in padding_shapes[1:]:
        if pad != padding_init:
            all_same = False
    return all_same, padding_init

# 3D卷积
class Conv3d(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(1, 1, 1),
                 stride=(1, 1, 1),
                 activation='relu',
                 padding='SAME',
                 use_bias=False,
                 use_bn=True):
        super(Conv3d,self).__init__()
        self.padding=padding
        self.activation=activation
        self.use_bias=use_bias
        self.use_bn=use_bn

        # padding
        if padding == 'SAME':
            padding_shape = get_padding_shape(kernel_size, stride)
            simplify_pad, pad_size = simplify_padding(padding_shape)
            self.simplify_pad = simplify_pad
        elif padding == 'VALID':
            padding_shape = 0
        else:
            raise ValueError(
                'padding should be in [VALID|SAME] but got {}'.format(padding))
        # 根据不同的padding方式
        if padding == 'SAME':
            if not simplify_pad:
                self.pad = torch.nn.ConstantPad3d(padding_shape, 0)
                self.conv3d = torch.nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    bias=use_bias)
            else:
                self.conv3d = torch.nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=pad_size,
                    bias=use_bias)
        elif padding == 'VALID':
            self.conv3d = torch.nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding_shape,  # padding shape=0
                stride=stride,
                bias=use_bias)
        else:
            raise ValueError(
                'padding should be in [VALID|SAME] but got {}'.format(padding))
        
        if use_bn:
            self.batch3d = torch.nn.BatchNorm3d(out_channels)


    def forward(self,inp):
        if self.padding=='SAME' and self.simplify_pad is False:
            inp=self.pad(inp)
        out=self.conv3d(inp)
        if self.use_bn:
            out = self.batch3d(out)
        if self.activation is not None:
            out = F.relu(out)
        return out

class Maxpool3d(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding='SAME'):
        super(Maxpool3d,self).__init__()
        if padding == 'SAME':
                padding_shape = get_padding_shape(kernel_size, stride)
                self.padding_shape = padding_shape
                self.pad = torch.nn.ConstantPad3d(padding_shape, 0)
        self.pool = torch.nn.MaxPool3d(kernel_size, stride, ceil_mode=True)     
        #ceil box=true 则计算数值时向上取整

    def forward(self, inp):
        inp = self.pad(inp)
        out = self.pool(inp)
        return out

class Inception(torch.nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Inception,self).__init__()
        # Branch 0
        self.branch_0 = Conv3d(
            in_channels, out_channels[0], kernel_size=(1, 1, 1))

        # Branch 1
        branch_1_conv1 = Conv3d(
            in_channels, out_channels[1], kernel_size=(1, 1, 1))
        branch_1_conv2 = Conv3d(
            out_channels[1], out_channels[2], kernel_size=(3, 3, 3))
        self.branch_1 = torch.nn.Sequential(branch_1_conv1, branch_1_conv2)

        # Branch 2
        branch_2_conv1 = Conv3d(
            in_channels, out_channels[3], kernel_size=(1, 1, 1))
        branch_2_conv2 = Conv3d(
            out_channels[3], out_channels[4], kernel_size=(3, 3, 3))
        self.branch_2 = torch.nn.Sequential(branch_2_conv1, branch_2_conv2)

        # Branch3
        branch_3_pool = Maxpool3d(
            kernel_size=(3, 3, 3), stride=(1, 1, 1), padding='SAME')
        branch_3_conv2 = Conv3d(
            in_channels, out_channels[5], kernel_size=(1, 1, 1))
        self.branch_3 = torch.nn.Sequential(branch_3_pool, branch_3_conv2)

    def forward(self,inp):
        out_0 = self.branch_0(inp)
        out_1 = self.branch_1(inp)
        out_2 = self.branch_2(inp)
        out_3 = self.branch_3(inp)
        out = torch.cat((out_0, out_1, out_2, out_3), 1)
        return out


class I3D(torch.nn.Module):
    def __init__(self,num_classes,modality='rgb',dropout_prob=0):
        super(I3D,self).__init__()
        self.num_classes = num_classes
        if modality == 'rgb':
            in_channels = 3
        elif modality == 'flow':
            in_channels = 2
        else:
            raise ValueError(
                '{} not among known modalities [rgb|flow]'.format(modality))
        self.modality = modality
        self.conv_1a7=Conv3d(
            out_channels=64,
            in_channels=in_channels,
            kernel_size=(7, 7, 7),
            stride=(2, 2, 2),
            padding='SAME')
        self.max_2a=Maxpool3d(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding='SAME')
        self.conv_2b1=Conv3d(
            out_channels=64,
            in_channels=64,
            kernel_size=(1, 1, 1),
            padding='SAME')
        self.conv_2c3=Conv3d(
            out_channels=192,
            in_channels=64,
            kernel_size=(3, 3, 3),
            padding='SAME')
        self.max_3a=Maxpool3d(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding='SAME')
        self.inception_3b=Inception(192, [64, 96, 128, 16, 32, 32])
        self.inception_3c=Inception(256, [128, 128, 192, 32, 96, 64])
        self.max_4a=Maxpool3d(
            kernel_size=(3, 3, 3), stride=(2, 2, 2), padding='SAME')
        self.inception_4b=Inception(480, [192, 96, 208, 16, 48, 64])
        self.inception_4c=Inception(512, [160, 112, 224, 24, 64, 64])
        self.inception_4d=Inception(512, [128, 128, 256, 24, 64, 64])
        self.inception_4e=Inception(512, [112, 144, 288, 32, 64, 64])
        self.inception_4f=Inception(528, [256, 160, 320, 32, 128, 128])
        self.max_5a2=Maxpool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding='SAME')
        self.inception_5b=Inception(832, [256, 160, 320, 32, 128, 128])
        self.inception_5c=Inception(832, [384, 192, 384, 48, 128, 128])
        self.avg_pool=torch.nn.AvgPool3d((2, 7, 7), (1, 1, 1))    # kernel size & stride
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.conv_6=Conv3d(
            in_channels=1024,
            out_channels=self.num_classes,
            kernel_size=(1, 1, 1),
            activation=None,
            use_bias=True,
            use_bn=False)
        self.softmax = torch.nn.Softmax(1)


    def forward(self,inp):
        out=self.conv_1a7(inp)
        out=self.max_2a(out)
        out=self.conv_2b1(out)
        out=self.conv_2c3(out)
        out=self.max_3a(out)
        out=self.inception_3b(out)
        out=self.inception_3c(out)
        out=self.max_4a(out)
        out=self.inception_4b(out)
        out=self.inception_4c(out)
        out=self.inception_4d(out)
        out=self.inception_4e(out)
        out=self.inception_4f(out)
        out=self.max_5a2(out)
        out=self.inception_5b(out)
        out=self.inception_5c(out)
        feature = self.avg_pool(out)
        out = self.dropout(feature)
        out=self.conv_6(feature)
        out = out.squeeze(3)
        out = out.squeeze(3)
        out,_ = out.max(2)          #没看懂这个在干嘛， max输出tensor的变量，为什么有,_
        out_logits = out
        out = self.softmax(out_logits)
        # out = self.sigmoid(out_logits) 
        return feature, out, out_logits