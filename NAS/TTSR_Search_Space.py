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
        self.loss_values={}

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

    # Input: loss_dict, sr, hr, is_init flag
    # Output: loss values
    # TODO: 1. trainer.vgg19 - pass in as parameter
    #       2. feat_list [S, T_lv3, T_lv2, T_lv1]
    def loss(self, sr, hr, loss_dict, is_init, vgg19, feat_dict):
        
        rec_loss = self.args.rec_w * loss_dict['rec_loss'](sr, hr)
        self.loss_values['rec_loss'] = rec_loss
        loss = self.loss_values['rec_loss']

        # calc perceptual loss
        if ('per_loss' in loss_dict):
            sr_relu5_1 = vgg19((sr + 1.) / 2.)
            with torch.no_grad():
                hr_relu5_1 = vgg19((hr.detach() + 1.) / 2.)
            per_loss = self.args.per_w * loss_dict['per_loss'](sr_relu5_1, hr_relu5_1)
            self.loss_values['per_loss'] = per_loss
            loss = loss + self.loss_values['per_loss']

        # calc tpl loss
        if ('tpl_loss' in loss_dict):
            sr_lv1, sr_lv2, sr_lv3 = self(sr=sr)
            tpl_loss = self.args.tpl_w * loss_dict['tpl_loss'](sr_lv3, sr_lv2, sr_lv1, 
                feat_dict['S'], feat_dict['T_lv3'], feat_dict['T_lv2'], feat_dict['T_lv1'])
            self.loss_values['tpl_loss'] = tpl_loss
            loss = loss + self.loss_values['tpl_loss']

        # calc adversarial loss
        if ('adv_loss' in loss_dict):
            adv_loss = self.args.adv_w * loss_dict['adv_loss'](sr, hr)
            self.loss_values['adv_loss'] = adv_loss
            loss = loss + self.loss_values['adv_loss']

        #calc arch loss
        if ('arch_loss' in loss_dict):
            arch_loss = loss_dict['arch_loss'](self)
            self.loss_values['arch_loss'] = arch_loss
            loss = loss + self.loss_values['arch_loss']
        
        return loss
    
    def get_loss_values(self):
        return self.loss_values

        

        



