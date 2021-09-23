
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
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(out_channels, out_channels)
        
    def forward(self, x):
        x1 = x
        out = self.conv1(x)
        out = self.relu(out.clone())
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
        
    def forward(self, x, feats_dict):
        x = F.relu(self.conv_head(x))
        x1 = x
        for i in range(self.num_res_blocks):
            x = self.RBs[i](x)
        x = self.conv_tail(x)
        x = x + x1
        feats_dict['x'] = x

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
        self.num_res_blocks = num_res_blocks
        self.ex12 = CSFI2(n_feats)
        self.RB21 = nn.ModuleList()
        self.RB22 = nn.ModuleList()
        for i in range(self.num_res_blocks):
            self.RB21.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                res_scale=res_scale))
            self.RB22.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                res_scale=res_scale))

        self.conv21_tail = conv3x3(n_feats, n_feats)
        self.conv22_tail = conv3x3(n_feats, n_feats)

    def forward(self,feats_dict):
        x22 = feats_dict['x22']
        x21 = feats_dict['x21']
        x22_res = x22
        ## cell 1-8 and 2-2
        x21_res, x22_res = self.ex12(feats_dict['x21_res'], x22_res)
        ## cell 1-10 2-4
        for i in range(self.num_res_blocks):
            x21_res = self.RB21[i](x21_res)
            x22_res = self.RB22[i](x22_res)
        ## 1-11 2-5
        x21_res = self.conv21_tail(x21_res)
        x22_res = self.conv22_tail(x22_res)
        ## 1-12 2-6
        x21 = x21 + x21_res
        x22 = x22 + x22_res
        feats_dict['x22'] = x22
        feats_dict['x21'] = x21

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
        self.num_res_blocks = num_res_blocks
        self.ex123 = CSFI3(n_feats)
        self.RB31 = nn.ModuleList()
        self.RB32 = nn.ModuleList()
        self.RB33 = nn.ModuleList()
        for i in range(self.num_res_blocks):
            self.RB31.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                res_scale=res_scale))
            self.RB32.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                res_scale=res_scale))
            self.RB33.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                res_scale=res_scale))

        self.conv31_tail = conv3x3(n_feats, n_feats)
        self.conv32_tail = conv3x3(n_feats, n_feats)
        self.conv33_tail = conv3x3(n_feats, n_feats)

    def forward(self, feats_dict):
        x31 = feats_dict['x31']
        x32 = feats_dict['x32']
        x33 = feats_dict['x33']
        x33_res = x33
        ## 1-13 2-7 4-2
        x31_res, x32_res, x33_res = self.ex123(feats_dict['x31_res'], feats_dict['x32_res'], x33_res)
        ## 1-15 2-9 4-4
        for i in range(self.num_res_blocks):
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

        feats_dict['x31'] = x31
        feats_dict['x32'] = x32
        feats_dict['x33'] = x33
        

#PS Module	    
class UpScaleModule(nn.Module):
    def __init__(self, n_feats):
        super(UpScaleModule, self).__init__()
        self.conv = conv3x3(n_feats, n_feats*4)
        self.ps = nn.PixelShuffle(2)

    def forward(self, feats_dict, stage):
        if stage == 2:
            x21 = feats_dict['x11']
            x21_res = x21
            x22 = self.conv(feats_dict['x11'])
            x22 = F.relu(self.ps(x22))
            feats_dict['x21'] = x21
            feats_dict['x21_res'] = x21_res
            feats_dict['x22'] = x22
        if stage == 3:
            x31 = feats_dict['x21']
            x31_res = x31
            x32 = feats_dict['x22']
            x32_res = x32
            x33 = self.conv(feats_dict['x22'])
            x33 = F.relu(self.ps(x33))
            feats_dict['x31'] = x31
            feats_dict['x31_res'] = x31_res
            feats_dict['x32'] = x32
            feats_dict['x32_res'] = x32_res
            feats_dict['x33'] = x33
            

class SAModule1x(nn.Module):
    def __init__(self, num_res_blocks, n_feats, res_scale):
        super(SAModule1x, self).__init__()
        self.num_res_blocks = num_res_blocks
        self.conv11_head = conv3x3(256+n_feats, n_feats)
        self.RB11 = nn.ModuleList()
        for i in range(self.num_res_blocks):
            self.RB11.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                res_scale=res_scale))
        self.conv11_tail = conv3x3(n_feats, n_feats)
    
    def forward(self, feats_dict, args_list):
        x11 = feats_dict['x']
        ### soft-attention
        x11_res = x11
        ### TT at cell 1-4
        x11_res = torch.cat((x11_res, args_list[1]), dim=1)
        x11_res = self.conv11_head(x11_res) #F.relu(self.conv11_head(x11_res))
        x11_res = x11_res * args_list[0]
        x11 = x11 + x11_res

        x11_res = x11
        ## 16 RB at cell 1-5
        for i in range(self.num_res_blocks):
            x11_res = self.RB11[i](x11_res)
        x11_res = self.conv11_tail(x11_res)
        
        ## cell 1-7
        x11 = x11 + x11_res
        feats_dict['x11'] = x11


class SAModule2x(nn.Module):
    def __init__(self, n_feats):
        super(SAModule2x, self).__init__()
        ### stage21, 22 - CSFI2 Module: Info exchange between 1x res & 2x res
        #self.conv21_head = conv3x3(n_feats, n_feats)
        self.conv22_head = conv3x3(128+n_feats, n_feats)
    
    def forward(self, feats_dict, args_list):
        ### soft-attention
        ## TT at cell 2-1
        x22 = feats_dict['x22']
        x22_res = x22
        x22_res = torch.cat((x22_res, args_list[2]), dim=1)
        ## 1-9 2-3
        x22_res = self.conv22_head(x22_res) #F.relu(self.conv22_head(x22_res))
        x22_res = x22_res * F.interpolate(args_list[0], scale_factor=2, mode='bicubic')
        x22 = x22 + x22_res

        feats_dict['x22'] = x22

class SAModule3x(nn.Module):
    def __init__(self, n_feats):
        super(SAModule3x, self).__init__()
        ### stage21, 22 - CSFI2 Module: Info exchange between 1x res & 2x res
        #self.conv31_head = conv3x3(n_feats, n_feats)
        #self.conv32_head = conv3x3(n_feats, n_feats)
        self.conv33_head = conv3x3(64+n_feats, n_feats)
    
    def forward(self, feats_dict, args_list):
        ### soft-attention
        ## 4-1
        x33 = feats_dict['x33']
        x33_res = x33
        x33_res = torch.cat((x33_res, args_list[3]), dim=1)
        ## 1-14 2-8 4-3
        x33_res = self.conv33_head(x33_res) #F.relu(self.conv33_head(x33_res))
        x33_res = x33_res * F.interpolate(args_list[0], scale_factor=4, mode='bicubic')
        x33 = x33 + x33_res

        feats_dict['x33'] = x33

class MixModule(nn.Module):
    def __init__(self, num_res_blocks, n_feats, res_scale, stage_num):
        super(MixModule, self).__init__()
        self.num_res_blocks = num_res_blocks
        if stage_num == 2:
            self.SAModule2x = SAModule2x(n_feats)
            self.CSFI2Module = CSFI2Module(num_res_blocks, n_feats, res_scale)
        if stage_num == 3:
            self.SAModule3x = SAModule3x(n_feats)
            self.CSFI3Module = CSFI3Module(num_res_blocks, n_feats, res_scale)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, feats_dict, args_list, weights, stage):
        SA_dict = feats_dict
        CSFI_dict = feats_dict
        if stage == 2:
            self.SAModule2x(SA_dict, args_list)
            self.CSFI2Module(CSFI_dict)
        if stage == 3:
            self.SAModule3x(SA_dict, args_list)
            self.CSFI3Module(CSFI_dict)

        weights = self.softmax(weights)
        if stage == 2:
            feats_dict['x22'] = weights[0]*SA_dict['x22']+weights[1]*CSFI_dict['x22']
            feats_dict['x21'] = weights[1]*CSFI_dict['x21']
        if stage == 3:
            feats_dict['x33'] = weights[0]*SA_dict['x33']+weights[1]*CSFI_dict['x33']
            feats_dict['x32'] = weights[1]*CSFI_dict['x32']
            feats_dict['x31'] = weights[1]*CSFI_dict['x31']
        
class MergeTail(nn.Module):
    def __init__(self, n_feats):
        super(MergeTail, self).__init__()
        self.conv13 = conv1x1(n_feats, n_feats)
        self.conv23 = conv1x1(n_feats, n_feats)
        self.conv_merge = conv3x3(n_feats*3, n_feats)
        self.conv_tail1 = conv3x3(n_feats, n_feats//2)
        self.conv_tail2 = conv1x1(n_feats//2, 3)

    def forward(self, feats_dict):
        x13 = F.interpolate(feats_dict['x31'], scale_factor=4, mode='bicubic')
        x13 = F.relu(self.conv13(x13))
        x23 = F.interpolate(feats_dict['x32'], scale_factor=2, mode='bicubic')
        x23 = F.relu(self.conv23(x23))

        x = F.relu(self.conv_merge( torch.cat((feats_dict['x33'], x13, x23), dim=1) ))
        x = self.conv_tail1(x)
        x = self.conv_tail2(x)
        x = torch.clamp(x, -1, 1)
        
        return x


class MainNet_NAS(nn.Module):
    def __init__(self, num_res_blocks, n_feats, res_scale, list_modules=[1,1]):
        super(MainNet_NAS, self).__init__()
        self.num_res_blocks = num_res_blocks ### a list containing number of resblocks of different stages
        self.list_modules = list_modules

        # Soft Feature Extraction Stage for 1x feat [necessary]
        self.SFE = SFE(self.num_res_blocks[0], n_feats, res_scale)

        # Soft Attention Module 1x -> see forward() for actual computation
        #self.stage1x = self.__make_stage1x(n_feats, res_scale) ### make stage 1x search space

        self.SAModule1x = SAModule1x(self.num_res_blocks[1], n_feats, res_scale)
        ### subpixel 1 -> 2 - PS Module: Upscaling 1x res to 2x res [necessary]
        self.ps12 = UpScaleModule(n_feats)

        # Soft Attention Module 2x -> see forward() for actual computation
        self.stage2x = self.__make_stage(n_feats, res_scale, 2) ### make stage 1x 2x search space

        ### subpixel 2 -> 3 - PS Module: Upscaling 2x res to 4x res [necessary]
        self.ps23 = UpScaleModule(n_feats)
        
        self.stage3x = self.__make_stage(n_feats, res_scale, 3) ### make stage 1x 2x 3x search space

        # Final Merge stage for 1x 2x 4x feats
        self.merge_tail = MergeTail(n_feats)

        self.__init_architecture()
        self.init_weights()

    
    def __make_stage(self, n_feats, res_scale, stage_num):
        modules = nn.ModuleList()
        if stage_num == 2:
            for _ in range(self.list_modules[0]):
                modules.append(MixModule(self.num_res_blocks[2], n_feats, res_scale, stage_num))
        if stage_num == 3:
            for _ in range(self.list_modules[1]):
                modules.append(MixModule(self.num_res_blocks[3], n_feats, res_scale, stage_num))
        return modules
    
    #def __make_stage1x(self, n_feats, res_scale):
    #    modules = nn.ModuleList()
    #    modules.append(SAModule1x(self.num_res_blocks, n_feats, res_scale))
    #    return modules
    
    #def __make_stage2x(self, n_feats, res_scale):
    #    modules = nn.ModuleList()
    #    modules.append(SAModule2x(self.num_res_blocks, n_feats, res_scale))
    #    modules.append(CSFI2Module(self.num_res_blocks, n_feats, res_scale))
    #    return modules
    
    #def __make_stage3x(self, n_feats, res_scale):
    #    modules = nn.ModuleList()
    #    modules.append(SAModule3x(self.num_res_blocks, n_feats, res_scale))
    #    modules.append(CSFI3Module(self.num_res_blocks, n_feats, res_scale))
    #    return modules
    def __init_architecture(self):
        self.arch_param = torch.randn(sum(self.list_modules[::]), 2, dtype=torch.float).cuda()*1e-3
        self.arch_param.requires_grad = True

    def arch_parameters(self):
        return [self.arch_param]

    def new(self):
        model_new = MainNet_NAS().cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy(y.data)
        return model_new

    def init_weights(self):
        print('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant(m.bias, 0)

    def forward(self, x, S=None, T_lv3=None, T_lv2=None, T_lv1=None):
        feats_dict = {}
        args_list = [S, T_lv3, T_lv2, T_lv1]
        ### shallow feature extraction (necessary)
        self.SFE(x, feats_dict)
        ### stage1x[0]: SAModule1x
        self.SAModule1x(feats_dict, args_list)
        ### Upscale from 1x -> 2x
        self.ps12(feats_dict,2)

        ### stage2x[0]: SAModule2x
        ### stage2x[1]: CSFI2Module
        idx = 0
        for i in range(self.list_modules[0]):
            self.stage2x[i](feats_dict, args_list, self.arch_param[idx+i],2)
        
        ### Upscale from 2x -> 3x
        self.ps23(feats_dict,3)
        ### stage3x[0]: SAModule3x
        ### stage3x[1]: CSFI3Module
        idx = idx + self.list_modules[1]
        for i in range(self.list_modules[0]):
            self.stage3x[i](feats_dict, args_list, self.arch_param[idx+i],3)
       
        ## 4-7 ~ 4-10
        x = self.merge_tail(feats_dict)

        return x
    
    
