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

    def forward(self, lr=None, lrsr=None, ref=None, refsr=None, sr=None):
        if (type(sr) != type(None)):
            ### used in transferal perceptual loss
            self.LTE_NAS_copy.load_state_dict(self.LTE_NAS.state_dict())
            sr_lv1, sr_lv2, sr_lv3 = self.LTE_NAS_copy((sr + 1.) / 2.)
            return sr_lv1, sr_lv2, sr_lv3

        _, _, lrsr_lv3  = self.LTE_NAS((lrsr.detach() + 1.) / 2.)
        _, _, refsr_lv3 = self.LTE_NAS((refsr.detach() + 1.) / 2.)

        ref_lv1, ref_lv2, ref_lv3 = self.LTE_NAS((ref.detach() + 1.) / 2.)

        S, T_lv3, T_lv2, T_lv1 = self.SearchTransfer_NAS(lrsr_lv3, refsr_lv3, ref_lv1, ref_lv2, ref_lv3)

        sr = self.MainNet_NAS(lr, S, T_lv3, T_lv2, T_lv1)

        return sr, S, T_lv3, T_lv2, T_lv1