from NAS import MainNet_NAS, LTE_NAS, SearchTransfer_NAS

import torch
import torch.nn as nn
import torch.nn.functional as F


class TTSR_Search_Space(nn.Module):
    def __init__(self, args):
        super(TTSR_Search_Space, self).__init__()
        self.args = args
        self.num_res_blocks = list( map(int, args.num_res_blocks.split('+')) )
        self.MainNet_NAS = MainNet_NAS.MainNet_NAS(num_res_blocks=self.num_res_blocks, n_feats=args.n_feats, res_scale=args.res_scale)
        self.LTE_NAS      = LTE_NAS.LTE_NAS(requires_grad=True)
        self.LTE_NAS_copy = LTE_NAS.LTE_NAS(requires_grad=False) ### used in transferal perceptual loss
        self.SearchTransfer_NAS = SearchTransfer_NAS.SearchTransfer_NAS()
        ### A way to initialize Architect Params
        self.__init_arch()
        ### A way to initialize Model Weights
        self.__init_weights()

        self.MSELoss = torch.nn.MSELoss().cuda()


    def __init_arch(self):
	### TODO
        self.arch_param=...
        self.arch_param.requires_grad = True

    def __init_weights(self):
	### TODO
        print('=> init weights from normal distribution')
        for m in self.MainNet_NAS.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
   
    def arch_parameters(self):
	    return [self.arch_param]

    def forward(self, lr=None, lrsr=None, ref=None, refsr=None, sr=None):
        if (type(sr) != type(None)):
            ### used in transferal perceptual loss
            self.LTE_copy.load_state_dict(self.LTE.state_dict())
            sr_lv1, sr_lv2, sr_lv3 = self.LTE_copy((sr + 1.) / 2.)
            return sr_lv1, sr_lv2, sr_lv3

        _, _, lrsr_lv3  = self.LTE((lrsr.detach() + 1.) / 2.)
        _, _, refsr_lv3 = self.LTE((refsr.detach() + 1.) / 2.)

        ref_lv1, ref_lv2, ref_lv3 = self.LTE((ref.detach() + 1.) / 2.)

        S, T_lv3, T_lv2, T_lv1 = self.SearchTransfer(lrsr_lv3, refsr_lv3, ref_lv1, ref_lv2, ref_lv3)

        sr = self.MainNet(lr, S, T_lv3, T_lv2, T_lv1)

        return sr, S, T_lv3, T_lv2, T_lv1

    def loss(self, input, target):
        logits = self(input)
        mse_loss = self.MSELoss(logits, target)
        arch_weights = torch.nn.functional.softmax(self.arch_param, dim=1)

        # regular_loss
        regular_loss = -arch_weights*torch.log10(arch_weights)-(1-arch_weights)*torch.log10(1-arch_weights)
        # latency loss: computing the average complexity (params num) of modules by running `python search_space.py`
        # latency_loss = torch.mean(arch_weights[:,0]*0.7+arch_weights[:,1]*0.3)

        return  mse_loss + regular_loss.mean()*0.01 #+ latency_loss*0.1


        loss = loss(clearer)*param1 + loss(ttsr)*param2