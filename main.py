from option import args
from utils import mkExpDir
from dataset import dataloader
from model import TTSR
from loss.loss import get_loss_dict
from trainer import Trainer

import os
import torch
import torch.nn as nn
import warnings
import deepspeed
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    ### make save_dir
    _logger = mkExpDir(args)

    ### dataloader of training set and testing set
    _dataloader = dataloader.get_dataloader(args) if (not args.test) else None
    _trainset = dataloader.get_trainset(args) if (not args.test) else None
    ### device and model
    #device = torch.device('cpu' if args.cpu else 'cuda')
    _model = TTSR.TTSR(args)
    #if ((not args.cpu) and (args.num_gpu > 1)):
        #_model = nn.DataParallel(_model, list(range(args.num_gpu)))

    ### loss
    ### {'rec_loss':ReconstructionLoss(), 
    ### 'per_loss':PerceptualLoss(), 
    ### 'tpl_loss':TPerceptualLoss(), 
    ### 'adv_loss':AdversarialLoss()}
    _loss_all = get_loss_dict(args, _logger)

    ### trainer
    t = Trainer(args, _logger, _dataloader, _model, _loss_all)

    ### DeepSpeed Initialization
    model_engine,optimizer,trainloader,_ = deepspeed.initialize(
        args=args,
        model=_model,
        optimizer=t.optimizer,
        training_data=_trainset,
        lr_scheduler=t.scheduler
    )

    ### test / eval / train
    if (args.test):
        t.load(model_path=args.model_path)
        t.test()
    elif (args.eval):
        t.load(model_path=args.model_path)
        t.evaluate()
    else:
        for epoch in range(1, args.num_init_epochs+1):
            t.train(model_engine, current_epoch=epoch, is_init=True)
        for epoch in range(1, args.num_epochs+1):
            t.train(model_engine, current_epoch=epoch, is_init=False)
            if (epoch % args.val_every == 0):
                t.evaluate(current_epoch=epoch)
