from option import args
from utils import mkExpDir
from dataset import dataloader
from NAS import TTSR_Search_Space
from NAS import TTSR_architect
from loss.loss import get_loss_dict
from trainer_NAS import Trainer

import os
import torch
import torch.nn as nn
import warnings
import math
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    ### make save_dir
    _logger = mkExpDir(args)

    ### dataloader of training set and testing set
    _dataloader = dataloader.get_dataloader(args) if (not args.test) else None

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    ### device and model
    device = torch.device('cpu' if args.cpu else 'cuda')
    
    _model = TTSR_Search_Space(args).to(device)
    _architect = TTSR_architect(_model, args)

    if ((not args.cpu) and (args.num_gpu > 1)):
        _model = nn.DataParallel(_model, list(range(args.num_gpu)))

    ### loss
    _loss_all = get_loss_dict(args, _logger)

    ### trainer
    t = Trainer(args, _logger, _dataloader, _model, _loss_all)

    ### test / eval / train
    if (args.test):
        t.load(model_path=args.model_path)
        t.test()
    elif (args.eval):
        t.load(model_path=args.model_path)
        t.evaluate()
    else:
        best_psnr = 0
        best_psnr_epoch = 0
        best_ssim = 0
        best_ssim_epoch = 0
        best_loss = float("inf") 
        best_loss_epoch = 0
        for epoch in range(1, args.num_init_epochs+1):
            # get current learning rate for model param training
            t.train(_architect, current_epoch=epoch, is_init=True)

        for epoch in range(1, args.num_epochs+1):
            # get current learning rate for model param training
            lr = t.scheduler.get_lr()
            t.train(current_epoch=epoch, is_init=False)
            psnr, ssim = t.infer(epoch)

            if psnr > best_psnr and not math.isinf(psnr):
                #utils.save(model, os.path.join(args.save, 'best_psnr_weights.pt'))
                best_psnr_epoch = epoch+1
                best_psnr = psnr
            if ssim > best_ssim:
                #utils.save(model, os.path.join(args.save, 'best_ssim_weights.pt'))
                best_ssim_epoch = epoch+1
                best_ssim = ssim
            
            t.scheduler.step()