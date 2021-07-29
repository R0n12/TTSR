
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=True)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=True)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, res_scale=1):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        
    def forward(self, x):
        x1 = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out * self.res_scale + x1
        return out

#Soft Feature Extraction 
class SFE(nn.Module):
    def __init__(self, num_res_blocks, n_feats, res_scale):
        super(SFE, self).__init__()
        self.num_res_blocks = num_res_blocks
        self.conv_head = conv3x3(3, n_feats)
        
        self.RBs = nn.ModuleList()
        for i in range(self.num_res_blocks):
            self.RBs.append(ResBlock(in_channels=n_feats, out_channels=n_feats, 
                res_scale=res_scale))
            
        self.conv_tail = conv3x3(n_feats, n_feats)
        
    def forward(self, x):
        x = F.relu(self.conv_head(x))
        x1 = x
        for i in range(self.num_res_blocks):
            x = self.RBs[i](x)
        x = self.conv_tail(x)
        x = x + x1
        return x

#CSFI Module for 1x & 2x
class CSFI2(nn.Module):
    def __init__(self, n_feats):
        super(CSFI2, self).__init__()
        self.conv12 = conv1x1(n_feats, n_feats)
        self.conv21 = conv3x3(n_feats, n_feats, 2)

        self.conv_merge1 = conv3x3(n_feats*2, n_feats)
        self.conv_merge2 = conv3x3(n_feats*2, n_feats)

    def forward(self, x1, x2):
        x12 = F.interpolate(x1, scale_factor=2, mode='bicubic')
        x12 = F.relu(self.conv12(x12))
        x21 = F.relu(self.conv21(x2))

        x1 = F.relu(self.conv_merge1( torch.cat((x1, x21), dim=1) ))
        x2 = F.relu(self.conv_merge2( torch.cat((x2, x12), dim=1) ))

        return x1, x2

#CSFI2 Module Wrapper
class CSFI2Module(nn.Module):
    def __init__(self, num_res_blocks, n_feats, res_scale):
        super(CSFI2Module, self).__init__()
        self.ex12 = CSFI2(n_feats)
        self.RB21 = nn.ModuleList()
        self.RB22 = nn.ModuleList()
        for i in range(self.num_res_blocks[2]):
            self.RB21.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                res_scale=res_scale))
            self.RB22.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                res_scale=res_scale))

        self.conv21_tail = conv3x3(n_feats, n_feats)
        self.conv22_tail = conv3x3(n_feats, n_feats)

    def forward(self, x22, x21, x21_res):
        x22_res = x22
        ## cell 1-8 and 2-2
        x21_res, x22_res = self.ex12(x21_res, x22_res)
        ## cell 1-10 2-4
        for i in range(self.num_res_blocks[2]):
            x21_res = self.RB21[i](x21_res)
            x22_res = self.RB22[i](x22_res)
        ## 1-11 2-5
        x21_res = self.conv21_tail(x21_res)
        x22_res = self.conv22_tail(x22_res)
        ## 1-12 2-6
        x21 = x21 + x21_res
        x22 = x22 + x22_res
        return x21, x22

#CSFI Module for 1x & 2x & 4x
class CSFI3(nn.Module):
    def __init__(self, n_feats):
        super(CSFI3, self).__init__()
        self.conv12 = conv1x1(n_feats, n_feats)
        self.conv13 = conv1x1(n_feats, n_feats)

        self.conv21 = conv3x3(n_feats, n_feats, 2)
        self.conv23 = conv1x1(n_feats, n_feats)

        self.conv31_1 = conv3x3(n_feats, n_feats, 2)
        self.conv31_2 = conv3x3(n_feats, n_feats, 2)
        self.conv32 = conv3x3(n_feats, n_feats, 2)

        self.conv_merge1 = conv3x3(n_feats*3, n_feats)
        self.conv_merge2 = conv3x3(n_feats*3, n_feats)
        self.conv_merge3 = conv3x3(n_feats*3, n_feats)

    def forward(self, x1, x2, x3):
        x12 = F.interpolate(x1, scale_factor=2, mode='bicubic')
        x12 = F.relu(self.conv12(x12))
        x13 = F.interpolate(x1, scale_factor=4, mode='bicubic')
        x13 = F.relu(self.conv13(x13))

        x21 = F.relu(self.conv21(x2))
        x23 = F.interpolate(x2, scale_factor=2, mode='bicubic')
        x23 = F.relu(self.conv23(x23))

        x31 = F.relu(self.conv31_1(x3))
        x31 = F.relu(self.conv31_2(x31))
        x32 = F.relu(self.conv32(x3))

        x1 = F.relu(self.conv_merge1( torch.cat((x1, x21, x31), dim=1) ))
        x2 = F.relu(self.conv_merge2( torch.cat((x2, x12, x32), dim=1) ))
        x3 = F.relu(self.conv_merge3( torch.cat((x3, x13, x23), dim=1) ))
        
        return x1, x2, x3

#CSFI3 Module Wrapper
class CSFI3Module(nn.Module):
    def __init__(self, num_res_blocks, n_feats, res_scale):
        super(CSFI3Module, self).__init__()
        self.ex123 = CSFI3(n_feats)

        self.RB31 = nn.ModuleList()
        self.RB32 = nn.ModuleList()
        self.RB33 = nn.ModuleList()
        for i in range(self.num_res_blocks[3]):
            self.RB31.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                res_scale=res_scale))
            self.RB32.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                res_scale=res_scale))
            self.RB33.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                res_scale=res_scale))

        self.conv31_tail = conv3x3(n_feats, n_feats)
        self.conv32_tail = conv3x3(n_feats, n_feats)
        self.conv33_tail = conv3x3(n_feats, n_feats)

    def forward(self, x33, x31_res, x32_res, x31, x32):
        x33_res = x33
        ## 1-13 2-7 4-2
        x31_res, x32_res, x33_res = self.ex123(x31_res, x32_res, x33_res)
        ## 1-15 2-9 4-4
        for i in range(self.num_res_blocks[3]):
            x31_res = self.RB31[i](x31_res)
            x32_res = self.RB32[i](x32_res)
            x33_res = self.RB33[i](x33_res)
        ## 1-16 2-10 4-5
        x31_res = self.conv31_tail(x31_res)
        x32_res = self.conv32_tail(x32_res)
        x33_res = self.conv33_tail(x33_res)
        ## 1-17
        x31 = x31 + x31_res
        ## 2-11
        x32 = x32 + x32_res
        ## 4-6
        x33 = x33 + x33_res
        
        return x31, x32, x33

#PS Module	    
class UpScaleModule(nn.Module):
    def __init__(self, n_feats):
        super(UpScaleModule, self).__init__()
        self.conv = conv3x3(n_feats, n_feats*4)
        self.ps = nn.PixelShuffle(2)

    def forward(self, x_low):
        x_high = self.conv(x_low)
        x_high = F.relu(self.ps(x_high))
        return x_high

class SAModule1x(nn.Module):
    def __init__(self, num_res_blocks, n_feats, res_scale):
        super(SAModule1x, self).__init__()
        self.conv11_head = conv3x3(256+n_feats, n_feats)
        self.RB11 = nn.ModuleList()
        for i in range(num_res_blocks[1]):
            self.RB11.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                res_scale=res_scale))
        self.conv11_tail = conv3x3(n_feats, n_feats)
    
    def forward(self, num_res_blocks, x, T_lv3, S):
        x11 = x
        ### soft-attention
        x11_res = x11
        ### TT at cell 1-4
        x11_res = torch.cat((x11_res, T_lv3), dim=1)
        x11_res = self.conv11_head(x11_res) #F.relu(self.conv11_head(x11_res))
        x11_res = x11_res * S
        x11 = x11 + x11_res

        x11_res = x11
        ## 16 RB at cell 1-5
        for i in range(num_res_blocks[1]):
            x11_res = self.RB11[i](x11_res)
        x11_res = self.conv11_tail(x11_res)
        
        ## cell 1-7
        x11 = x11 + x11_res

        return x11

class SAModule2x(nn.Module):
    def __init__(self, num_res_blocks, n_feats, res_scale):
        super(SAModule1x, self).__init__()
        ### stage21, 22 - CSFI2 Module: Info exchange between 1x res & 2x res
        #self.conv21_head = conv3x3(n_feats, n_feats)
        self.conv22_head = conv3x3(128+n_feats, n_feats)
    
    def forward(self, num_res_blocks, x22, T_lv2, S):
        ### soft-attention
        ## TT at cell 2-1
        x22_res = x22
        x22_res = torch.cat((x22_res, T_lv2), dim=1)
        ## 1-9 2-3
        x22_res = self.conv22_head(x22_res) #F.relu(self.conv22_head(x22_res))
        x22_res = x22_res * F.interpolate(S, scale_factor=2, mode='bicubic')
        x22 = x22 + x22_res

        return x22

class SAModule3x(nn.Module):
    def __init__(self, num_res_blocks, n_feats, res_scale):
        super(SAModule1x, self).__init__()
        ### stage21, 22 - CSFI2 Module: Info exchange between 1x res & 2x res
        #self.conv31_head = conv3x3(n_feats, n_feats)
        #self.conv32_head = conv3x3(n_feats, n_feats)
        self.conv33_head = conv3x3(64+n_feats, n_feats)
    
    def forward(self, num_res_blocks, x33, T_lv1, S):
        ### soft-attention
        ## 4-1
        x33_res = x33
        x33_res = torch.cat((x33_res, T_lv1), dim=1)
        ## 1-14 2-8 4-3
        x33_res = self.conv33_head(x33_res) #F.relu(self.conv33_head(x33_res))
        x33_res = x33_res * F.interpolate(S, scale_factor=4, mode='bicubic')
        x33 = x33 + x33_res

        return x33

        
class MergeTail(nn.Module):
    def __init__(self, n_feats):
        super(MergeTail, self).__init__()
        self.conv13 = conv1x1(n_feats, n_feats)
        self.conv23 = conv1x1(n_feats, n_feats)
        self.conv_merge = conv3x3(n_feats*3, n_feats)
        self.conv_tail1 = conv3x3(n_feats, n_feats//2)
        self.conv_tail2 = conv1x1(n_feats//2, 3)

    def forward(self, x1, x2, x3):
        x13 = F.interpolate(x1, scale_factor=4, mode='bicubic')
        x13 = F.relu(self.conv13(x13))
        x23 = F.interpolate(x2, scale_factor=2, mode='bicubic')
        x23 = F.relu(self.conv23(x23))

        x = F.relu(self.conv_merge( torch.cat((x3, x13, x23), dim=1) ))
        x = self.conv_tail1(x)
        x = self.conv_tail2(x)
        x = torch.clamp(x, -1, 1)
        
        return x


class MainNet_NAS(nn.Module):
    def __init__(self, num_res_blocks, n_feats, res_scale):
        super(MainNet_NAS, self).__init__()
        self.num_res_blocks = num_res_blocks ### a list containing number of resblocks of different stages
        self.n_feats = n_feats

        # Soft Feature Extraction Stage for 1x feat [necessary]
        self.SFE = SFE(self.num_res_blocks[0], n_feats, res_scale)

        # Soft Attention Module 1x -> see forward() for actual computation
        self.stage1x = self.__make_stage1x(num_res_blocks, n_feats, res_scale) ### make stage 1x search space

        ### subpixel 1 -> 2 - PS Module: Upscaling 1x res to 2x res [necessary]
        self.ps12 = UpScaleModule(n_feats)

        # Soft Attention Module 2x -> see forward() for actual computation
        self.stage2x = self.__make_stage2x(num_res_blocks, n_feats, res_scale) ### make stage 1x 2x search space

        ### subpixel 2 -> 3 - PS Module: Upscaling 2x res to 4x res [necessary]
        self.ps23 = UpScaleModule(n_feats)
        
        self.stage3x = self.__make_stage3x(num_res_blocks, n_feats, res_scale) ### make stage 1x 2x 3x search space

        # Final Merge stage for 1x 2x 4x feats
        self.merge_tail = MergeTail(n_feats)
    
    def __make_stage1x(self, num_res_blocks, n_feats, res_scale):
        modules = nn.ModuleList()
        modules.append(SAModule1x(num_res_blocks, n_feats, res_scale))
        return modules
    
    def __make_stage2x(self, num_res_blocks, n_feats, res_scale):
        modules = nn.ModuleList()
        modules.append(SAModule2x(num_res_blocks, n_feats, res_scale))
        modules.append(CSFI2Module(num_res_blocks, n_feats, res_scale))
        return modules
    
    def __make_stage3x(self, num_res_blocks, n_feats, res_scale):
        modules = nn.ModuleList()
        modules.append(SAModule3x(num_res_blocks, n_feats, res_scale))
        modules.append(CSFI3Module(num_res_blocks, n_feats, res_scale))
        return modules


    def forward(self, num_res_blocks, x, S=None, T_lv3=None, T_lv2=None, T_lv1=None):
        ### shallow feature extraction (necessary)
        x = self.SFE(x)
        ### stage1x[0]: SAModule1x
        x11 = self.stage1x[0](num_res_blocks, x, T_lv3, S)

        x21 = x11
        x21_res = x21
        ### Upscale from 1x -> 2x
        x22 = self.ps12(x11)
        ### stage2x[0]: SAModule2x
        ### stage2x[1]: CSFI2Module
        x22 = self.stage2x[0](num_res_blocks, x22, T_lv2, S)
        x21, x22 = self.stage2x[1](x22, x21, x21_res)

        x31 = x21
        x31_res = x31
        x32 = x22
        x32_res = x32
        ### Upscale from 2x -> 3x
        x33 = self.ps23(x22)
        ### stage3x[0]: SAModule3x
        ### stage3x[1]: CSFI3Module
        x33 = self.stage3x[0](num_res_blocks, x33, T_lv1, S)
        x31, x32, x33 = self.stage3x[1](x33, x31_res, x32_res, x31, x32)
       
        ## 4-7 ~ 4-10
        x = self.merge_tail(x31, x32, x33)

        return x
