from option import parser
from utils import mkExpDir
from utils import save
from dataset import dataloader
from NAS import TTSR_Search_Space
from NAS import TTSR_Architect
from loss.loss import get_loss_dict
from trainer_NAS import Trainer

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

import os
import torch
import torch.nn as nn
import warnings
import math
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    args = parser.parse_args()
    ### make save_dir
    _logger = mkExpDir(args)

    ### dataloader of training set and testing set
    _dataloader = dataloader.get_dataloader(args) if (not args.test) else None

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    ### device and model
    device = torch.device('cpu' if args.cpu else 'cuda')
    
    _model = TTSR_Search_Space.TTSR_Search_Space(args).to(device)
    _architect = 0
    if args.NAS: _architect = TTSR_Architect.TTSR_Architect(_model, args)

    if ((not args.cpu) and (args.num_gpu > 1)):
        _model = nn.DataParallel(_model, list(range(args.num_gpu)))

    ### loss
    _loss_all = get_loss_dict(args, _logger)

    ### trainer
    t = Trainer(args, _logger, _dataloader, _model, _architect, _loss_all)

    ### test / eval / train
    if (args.test):
        t.load(model_path=args.model_path)
        t.test()
    elif (args.eval):
        t.load(model_path=args.model_path)
        t.evaluate()
    elif not args.NAS:
        print("Default TTSR training ...\n")
        for epoch in range(1, args.num_init_epochs+1):
            t.train(current_epoch=epoch, is_init=True)
        for epoch in range(1, args.num_epochs+1):
            t.train(current_epoch=epoch, is_init=False)
            if (epoch % args.val_every == 0):
                t.evaluate(current_epoch=epoch)
    else:
        best_psnr = 0
        best_psnr_epoch = 0
        best_ssim = 0
        best_ssim_epoch = 0
        best_loss = float("inf") 
        best_loss_epoch = 0
        for epoch in range(args.num_init_epochs):
            print("Current init epoch: ", epoch+1)
            # get current learning rate for model param training
            t.train(_architect, current_epoch=epoch, is_init=True)


        for epoch in range(args.num_epochs):
            # get current learning rate for model param training
            lr = t.scheduler.get_lr()
            t.train_NAS(_architect, current_epoch=epoch, is_init=False)
            psnr, ssim = t.infer(epoch)

            if psnr > best_psnr and not math.isinf(psnr):
                save(_model, os.path.join(args.save_dir, 'best_psnr_weights.pt'))
                best_psnr_epoch = epoch+1
                best_psnr = psnr
            if ssim > best_ssim:
                save(_model, os.path.join(args.save_dir, 'best_ssim_weights.pt'))
                best_ssim_epoch = epoch+1
                best_ssim = ssim
            
            t.logger.info('psnr:%6f ssim:%6f -- best_psnr:%6f best_ssim:%6f', psnr, ssim, best_psnr, best_ssim)
            t.logger.info('Current arch_parameters: %s', _model.MainNet_NAS.arch_parameters())
            t.logger.info('arch:%s', torch.argmax(_model.MainNet_NAS.arch_parameters()[0], dim=1)) 
            t.scheduler.step()

    t.logger.info('BEST_LOSS(epoch):%6f(%d), BEST_PSNR(epoch):%6f(%d), BEST_SSIM(epoch):%6f(%d)', best_loss, best_loss_epoch, best_psnr, best_psnr_epoch, best_ssim, best_ssim_epoch)
    save(_model, os.path.join(args.save_dir, 'last_weights.pt'))
