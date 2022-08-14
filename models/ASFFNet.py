# -*- coding: utf-8 -*-
import torch
from math import sqrt
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as SpectralNorm
from utils.ROI_MLS import roi_mls_whole_final

import warnings
warnings.filterwarnings('ignore')

##########################################################################################
###############ASFFNet re-implemented by Xiaoming Li for 512*512 image
##########################################################################################

class ASFFNet(nn.Module):
    def __init__(self):
        super(ASFFNet, self).__init__()
        model = []
        SFF_Num = 4 #
        for i in range(SFF_Num):
            model.append(ASFFBlock(128))
        model.append(SFFLayer_last(128))
        self.sff_branch = nn.Sequential(*model)
        self.MSDilate = MSDilateBlock(128, dilation = [4,3,2,1])  
      
        self.UpModel = nn.Sequential( #
            nn.Upsample(scale_factor = 2,mode='bilinear',align_corners=True), # 
            SpectralNorm(nn.Conv2d(128, 128, 3, 1, 1)),
            nn.LeakyReLU(0.2, True),
            UpDilateResBlock(128, [1,2]),
            SpectralNorm(nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2, True),
            UpDilateResBlock(64, [1,2]),
            nn.Upsample(scale_factor = 2,mode='bilinear',align_corners=True), # 
            UpDilateResBlock(64, [1,1]),
            SpectralNorm(nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)),
            nn.Tanh()
        )
        self.LQ_Model = FeatureExDilatedResNet(64)
        self.Ref_Model = FeatureExDilatedResNet(64)
        self.Mask_CModel = MaskConditionModel(128)
        self.Component_CModel = ComponentConditionModel(128)  #


    def forward(self, LQ, Ref, Mask, lq_landmark, ref_landmark):
        LQ_Feature = self.LQ_Model(LQ) #
        Ref_Feature = self.Ref_Model(Ref)#
        
        DownScale = LQ.size(2) / LQ_Feature.size(2)
        lq_landmark_D = lq_landmark / DownScale
        ref_landmark_D = ref_landmark / DownScale

        Grids = torch.zeros(Ref_Feature.size(0),Ref_Feature.size(2),Ref_Feature.size(3),2).type_as(Ref_Feature)
        for i in range(LQ_Feature.size(0)):
            # mls
            Grids[i,:,:,:] = roi_mls_whole_final(Ref_Feature[i, :, :, :], lq_landmark_D[i, 17:, :], ref_landmark_D[i, 17:, :])

        Ref_AdaIn = adaptive_instance_normalization_4D(Ref_Feature, LQ_Feature)
        MLS_Ref_Feature = F.grid_sample(Ref_AdaIn, Grids, mode='bilinear') #
        MaskC = self.Mask_CModel(Mask) #
        ComponentC = self.Component_CModel(MLS_Ref_Feature) #

        Fea = self.sff_branch((LQ_Feature, ComponentC, MaskC))  #
        MSFea = self.MSDilate(Fea)
        out = self.UpModel(MSFea + LQ_Feature)

        return out

        
class FeatureExDilatedResNet(nn.Module):
    def __init__(self, ngf = 64):
        super(FeatureExDilatedResNet, self).__init__()
        self.conv1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(3, ngf, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            )
        self.stem1_1 = resnet_block(ngf, dilation=[7,5])
        self.stem1_2 = resnet_block(ngf, dilation=[7,5])

        self.conv2 = nn.Sequential(
            SpectralNorm(nn.Conv2d(ngf, ngf*2, 3, 2, 1)),
            nn.LeakyReLU(0.2),
            )
        self.stem2_1 = resnet_block(ngf*2, dilation=[5,3])
        self.stem2_2 = resnet_block(ngf*2, dilation=[5,3])

        self.conv3 = nn.Sequential(
            SpectralNorm(nn.Conv2d(ngf*2, ngf*4, 3, 2, 1)),
            nn.LeakyReLU(0.2),
            )

        self.stem3_1 = resnet_block(ngf*4, dilation=[3,1])
        self.stem3_2 = resnet_block(ngf*4, dilation=[3,1])

        self.conv4 = nn.Sequential(
            SpectralNorm(nn.Conv2d(ngf*4, ngf*2, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            )

    def forward(self, img): #
        fea1 = self.stem1_2(self.stem1_1(self.conv1(img)))
        fea2 = self.stem2_2(self.stem2_1(self.conv2(fea1)))
        fea3 = self.stem3_2(self.stem3_1(self.conv3(fea2)))
        fea4 = self.conv4(fea3)
        return fea4


class MaskConditionModel(nn.Module):
    def __init__(self, out_channels):
        super(MaskConditionModel,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, 9, 2, 4, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, 7, 1, 3, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 5, 2, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, out_channels, 3, 1, 1, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        return self.model(x) #

def MSconvU(in_channels, out_channels,conv_layer, kernel_size=3, stride=1,dilation=1, bias=True):
    return nn.Sequential(
        SpectralNorm(conv_layer(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=((kernel_size-1)//2)*dilation, bias=bias)),
        nn.LeakyReLU(0.2)
    )

class MSDilateBlock(nn.Module):#
    def __init__(self, in_channels,conv_layer=nn.Conv2d, kernel_size=3, dilation=[7,5,3,1], bias=True):
        super(MSDilateBlock, self).__init__()
        self.conv1 =  MSconvU(in_channels, in_channels//2,conv_layer, kernel_size,dilation=dilation[0], bias=bias)
        self.conv2 =  MSconvU(in_channels, in_channels//2,conv_layer, kernel_size,dilation=dilation[1], bias=bias)
        self.conv3 =  MSconvU(in_channels, in_channels//2,conv_layer, kernel_size,dilation=dilation[2], bias=bias)
        self.conv4 =  MSconvU(in_channels, in_channels//2,conv_layer, kernel_size,dilation=dilation[3], bias=bias)
        self.convi =  nn.Sequential(
            SpectralNorm(conv_layer(in_channels*2, in_channels*1, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2, bias=bias)),
            nn.LeakyReLU(0.2),
            SpectralNorm(conv_layer(in_channels*1, in_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2, bias=bias)),
        )
        
    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        cat  = torch.cat([conv1, conv2, conv3, conv4], 1)
        out = self.convi(cat) + x
        return out
        

class UpResBlock(nn.Module):
    def __init__(self, dim):
        super(UpResBlock, self).__init__()
        self.Model = nn.Sequential(
            SpectralNorm(nn.Conv2d(dim, dim, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(dim, dim, 3, 1, 1)),
            nn.LeakyReLU(0.2),
        )
    def forward(self, x):
        out = x + self.Model(x)
        return out

class ComponentConditionModel(nn.Module):
    def __init__(self,in_channel):
        super(ComponentConditionModel, self).__init__()
        self.model = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channel, in_channel, 1)),
            nn.LeakyReLU(0.2),
            UpResBlock(in_channel),
            SpectralNorm(nn.Conv2d(in_channel, in_channel, 1)),
            nn.LeakyReLU(0.2),
            UpResBlock(in_channel),
            SpectralNorm(nn.Conv2d(in_channel, in_channel, 1)),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.model(x) #


class SFFLayer(nn.Module):
    def __init__(self, in_channels):
        super(SFFLayer, self).__init__()
        self.MaskModel = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels //2 * 3, in_channels, 3, 1, 1, bias=False)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
        )
        self.MaskConcat = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels, in_channels //2 , 1, 1)),
        )
        self.DegradedModel = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels, in_channels, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(in_channels, in_channels, 3, 1, 1)),
        )
        self.DegradedConcat = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels, in_channels //2 , 1, 1)),
        )
        self.RefModel = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels,in_channels,3, 1, 1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(in_channels, in_channels, 3, 1, 1)),
        )
        self.RefConcat = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels, in_channels //2 , 1, 1)),
        )

    def forward(self, X):
        #X[0] feature, X[1]: Ref Feature, X[2]: MaskFeature
        MaskC = X[2]
        DegradedF = self.DegradedModel(X[0])
        RefF = self.RefModel(X[1])

        DownMask = self.MaskConcat(MaskC)
        DownDegraded = self.DegradedConcat(DegradedF)
        DownRef = self.RefConcat(RefF)

        ConcatMask = torch.cat((DownMask,DownDegraded,DownRef),1)
        MaskF = self.MaskModel(ConcatMask)

        return DegradedF + (RefF - DegradedF) * MaskF

class SFFLayer_last(nn.Module):
    def __init__(self, in_channels):
        super(SFFLayer_last, self).__init__()
        self.MaskModel = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels //2 * 3, in_channels, 3, 1, 1, bias=False)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False)),
        )
        self.MaskConcat = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels, in_channels //2 , 1, 1)),
        )
        self.DegradedModel = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels, in_channels, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(in_channels, in_channels, 3, 1, 1)),
        )
        self.DegradedConcat = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels, in_channels //2 , 1, 1)),
        )
        self.RefModel = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels,in_channels,3, 1, 1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(in_channels, in_channels, 3, 1, 1)),
        )
        self.RefConcat = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels, in_channels //2 , 1, 1)),
        )

    def forward(self, X):
        #X[0] LQ feature, X[1]: Ref Feature, X[2]: MaskFeature
        MaskC = X[2]
        DegradedF = self.DegradedModel(X[0])
        RefF = self.RefModel(X[1])

        DownMask = self.MaskConcat(MaskC)
        DownDegraded = self.DegradedConcat(DegradedF)
        DownRef = self.RefConcat(RefF)

        ConcatMask = torch.cat((DownMask,DownDegraded,DownRef),1)
        MaskF = self.MaskModel(ConcatMask)
        return DegradedF + (RefF - DegradedF) * MaskF


class ASFFBlock(nn.Module):
    def __init__(self, in_channels=128):
        super(ASFFBlock, self).__init__()
        self.sff0 = SFFLayer(in_channels)
        self.conv0 = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels, in_channels, 5, 1, 2)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(in_channels, in_channels, 3, 1, 1)),
            nn.LeakyReLU(0.2)
        )
        self.sff1 = SFFLayer(in_channels)
        self.conv1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels, in_channels, 5, 1, 2)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(in_channels, in_channels, 3, 1, 1)),
            nn.LeakyReLU(0.2)
        )

    def forward(self, X):
        fea1_1 = self.sff0(X)
        fea1_2 = self.conv0(fea1_1)
        fea2_1 = self.sff1((fea1_2, X[1], X[2]))
        fea2_2 = self.conv1(fea2_1)
        return (X[0] + fea2_2, X[1], X[2])


class UpDilateResBlock(nn.Module):
    def __init__(self, dim, dilation=[2,1] ):
        super(UpDilateResBlock, self).__init__()
        self.Model0 = nn.Sequential(
            SpectralNorm(nn.Conv2d(dim, dim, 3, 1, ((3-1)//2)*dilation[0], dilation[0])),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(dim, dim, 3, 1, ((3-1)//2)*dilation[0], dilation[0])),
            nn.LeakyReLU(0.2),
        )
        self.Model1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(dim, dim, 3, 1, ((3-1)//2)*dilation[1], dilation[1])),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(dim, dim, 3, 1, ((3-1)//2)*dilation[1], dilation[1])),
            nn.LeakyReLU(0.2),
        )
    def forward(self, x):
        out = x + self.Model0(x)
        out2 = out + self.Model1(out)
        return out2

def adaptive_instance_normalization_4D(content_feat, style_feat): #
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std_4D(style_feat)
    content_mean, content_std = calc_mean_std_4D(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def calc_mean_std_4D(feat, eps=1e-5):
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def resnet_block(in_channels, conv_layer = nn.Conv2d, norm_layer = nn.BatchNorm2d, kernel_size = 3, dilation = [1,1], bias=True):
    return ResnetBlock(in_channels,conv_layer, norm_layer, kernel_size, dilation, bias=bias)

class ResnetBlock(nn.Module):
    def __init__(self, in_channels,conv_layer = nn.Conv2d, norm_layer = nn.BatchNorm2d, kernel_size = 3, dilation = [1,1], bias=True):
        super(ResnetBlock, self).__init__()
        self.stem = nn.Sequential(
            SpectralNorm(conv_layer(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[0], padding=((kernel_size-1)//2)*dilation[0], bias=bias)),
            nn.LeakyReLU(0.2),
            SpectralNorm(conv_layer(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[1], padding = ((kernel_size-1)//2)*dilation[1], bias=bias)),
        )
    def forward(self, x):
        out = self.stem(x) + x
        return out

if __name__ == '__main__':
    print('Test')
