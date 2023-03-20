import os
import sys
import random
import math
import time
import pathlib
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
from config import parse_arguments
from datasets import DiseaseDataset

from models.resnet import resnet18, resnet50, resnet152
from utils_folder.utils import AverageMeter, ProgressMeter, save_model

import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from tensorboardX import SummaryWriter


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self. count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        
        print('\r' + '\t'.join(entries), end = '')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

import torch

def save_model(name, epoch, model, optimizer,scheduler):
    torch.save({
        'epoch' : epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'scheduler' : scheduler.state_dict()
        }, name)


def calculate_parameters(model):
    return sum(param.numel() for param in model.parameters())/1000000.0

def train(args, epoch, loader, val_loader, model, device, optimizer, writer ,scheduler):
    print('[*] Train Phase')

    model.train()
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    accuracy = AverageMeter('ACC', ':2.4f')
    progress = ProgressMeter(
        len(loader),
        [batch_time, losses, accuracy],
        prefix='Epoch: [{}]'.format(epoch))

    correct = 0
    total = 0
    overall_logits = []
    end = time.time()
    running_loss = 0
    for iter_, (imgs, labels) in enumerate(iter(loader)):
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        outputs = model(imgs)

        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(outputs, labels)
        '''
        have to modify multi label acc
        '''
        outputs = torch.sigmoid(outputs)
        outputs_preds = outputs.clone()
        overall_logits += outputs.cpu().detach().numpy().tolist()

        outputs_preds[outputs_preds >= 0.5] = 1
        outputs_preds[outputs_preds < 0.5] = 0
        total_ = labels.size(0) * labels.size(1)
        correct_ = torch.sum(outputs_preds == labels.data).item()
        total += total_
        correct += correct_

        accuracy.update(100*correct_/total_)
        losses.update(loss.item(), imgs[0].size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        progress.display(iter_)

        if (iter_ % args.print_freq == 0)& (iter_ != 0):
            print()
            writer.add_scalar('train_loss', running_loss/iter_, (epoch*len(loader))+iter_)
            writer.add_scalar('train_acc', 100.*correct/total, (epoch*len(loader))+iter_)
            
    print()
    print('[*] Valid Phase')
    model.eval()

    val_batch_time = AverageMeter('Time', ':6.3f')
    val_losses = AverageMeter('Loss', ':.4f')
    val_accuracy = AverageMeter('ACC', ':2.4f')
    progress = ProgressMeter(
        len(val_loader),
        [val_batch_time, val_losses, val_accuracy],
        prefix='Epoch: [{}]'.format(epoch))

    val_correct = 0
    val_total = 0
    overall_logits = []
    val_running_loss = 0
    
    mutant_correct = 0
    mutant_total = 0
    wild_correct = 0
    wild_total = 0

    end = time.time()

    with torch.no_grad():
        for iter_, (imgs, labels) in enumerate(iter(val_loader)):
            
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)

            criterion = nn.BCEWithLogitsLoss()
            loss = criterion(outputs, labels)
            '''
            have to modify multi label acc
            '''
            outputs = torch.sigmoid(outputs)
            outputs_preds = outputs.clone()
            overall_logits += outputs.cpu().detach().numpy().tolist()

            outputs_preds[outputs_preds >= 0.5] = 1
            outputs_preds[outputs_preds < 0.5] = 0
            total_ = labels.size(0) * labels.size(1)
            correct_ = torch.sum(outputs_preds == labels.data).item()
            val_total += total_
            val_correct += correct_
            
            
            wild_total_ = torch.sum(labels.data).item()
            mutant_total_ = total_ - wild_total_
            
            mutant_total += mutant_total_
            wild_total += wild_total_
            
            wild_correct_ = torch.sum(outputs_preds * labels).item()
            mutant_correct_ = correct_ - wild_correct_
            
            mutant_correct += mutant_correct_
            wild_correct += wild_correct_
            
            
            val_accuracy.update(100*correct_/total_)
            val_losses.update(loss.item(), imgs[0].size(0))
            val_batch_time.update(time.time() - end)
            end = time.time()
            val_running_loss += loss.item()
            
            progress.display(iter_)

        
        writer.add_scalar('val_loss', val_running_loss/iter_, epoch)
        writer.add_scalar('val_acc', 100.*val_correct/val_total, epoch)
        writer.add_scalar('val_mutant_acc', 100.*mutant_correct/mutant_total, epoch)
        writer.add_scalar('val_wild_acc', 100.*wild_correct/wild_total, epoch)
        
        scheduler.step()
    model.train()
    print()



def test(args, epoch, loader, model, device, writer):

    print('[*] Test Phase')
    model.eval()
    correct = 0
    total = 0
    overall_logits = []
    overall_preds = []
    overall_gts = []

    with torch.no_grad():
        for iter_, (imgs, labels) in enumerate(iter(loader)):
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            outputs = torch.sigmoid(outputs)
            outputs_preds = outputs.clone()
            overall_logits += outputs.cpu().detach().numpy().tolist()

            outputs_preds[outputs_preds >= 0.5] = 1
            outputs_preds[outputs_preds < 0.5] = 0
            total += labels.size(0) * labels.size(1)
            correct += torch.sum(outputs_preds == labels.data).item()
        
    test_acc = 100.*correct/total
    print('[*] Test Acc: {:5f}'.format(test_acc))
    writer.add_scalar('Test acc', test_acc, epoch)

    model.train()
    return test_acc




def main(args):
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

    ##### Initial Settings
#     csv_data = pd.read_csv(args.csv_file)
#     class_list = csv_data.keys().tolist()[5:] # warning

    class_list = ['IDH_Mutant', 'IDH_Wild']
    print("[*] class list : " , class_list)
    num_classes = args.num_class
    downstream = '{}_{}_class'.format(args.downstream_name, num_classes)

    print('\n[*****] ', downstream)
    print('[*] using {} bit images'.format(args.bit))

    # device check & pararrel
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('[*] device: ', device)

    # path setting
#     pathlib.Path(args.log_dir).mkdir(parents=True, exist_ok=True)
#     pathlib.Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    num = args.num
    print(num)
    
    
    backbone = args.backbone
    folder_name = 'experiment_{}/{}/{}_{:03}'.format(num, backbone, backbone, args.fake_slice)
    
    if args.clinical != 'no':
        folder_name += f'_{args.clinical}'

    args.checkpoint_dir = os.path.join(folder_name, args.checkpoint_dir)
    
    args.checkpoint_dir = os.path.join('experiment_for_revision', args.checkpoint_dir)
    args.log_dir        = os.path.join('experiment_for_revision', args.log_dir)

    pathlib.Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # for log
    f = open(os.path.join(args.log_dir,'arguments.txt'), 'w')
    f.write(str(args))
    f.close()
    print('[*] log directory: {} '.format(args.log_dir))
    print('[*] checkpoint directory: {} '.format(args.checkpoint_dir))
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed) 
        np.random.seed(args.seed) 
        torch.cuda.manual_seed(args.seed) 
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False 

    # select network
    print('[*] build network... backbone: {}'.format(args.backbone))
    if args.backbone == 'resnet50':
        model = resnet50(num_classes=args.num_class)
    if args.backbone == 'resnet18':
        model = resnet18(num_classes=args.num_class)
    elif args.backbone == 'vgg':
        model = vgg16(num_classes=args.num_class)
    elif args.backbone == 'densenet':
        model = densenet169(num_classes=args.num_class)
    elif args.backbone == 'inception':
        model = Inception3(num_classes=args.num_class)
    else:
        ValueError('Have to set the backbone network in [resnet, vgg, densenet]')

    print(('[*] Total Params: %.2fM'%(calculate_parameters(model))))

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)


    writer = SummaryWriter(args.log_dir)    

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay = args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.99 ** epoch)

    if args.resume is True:
        checkpoint = torch.load(args.pretrained)
        pretrained_dict = checkpoint['state_dict']
        pretrained_dict = {key.replace("module.", ""): value for key, value in pretrained_dict.items()}
        model.load_state_dict(pretrained_dict)
        args.start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    model = model.to(device)
    
    
    ##### Dataset & Dataloader
    print('[*] prepare datasets & dataloader...')

    train_datasets = DiseaseDataset(args, mode = 'train', clinical = args.clinical)
    val_datasets = DiseaseDataset(args, mode = 'valid')
    test_datasets = DiseaseDataset(args, mode = 'internal_test')
        
    print(len(train_datasets))

    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=args.batch_size, 
                                num_workers=args.w, pin_memory=False, 
                                shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_datasets, batch_size=args.batch_size, 
                                num_workers=args.w, pin_memory=False, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=1, 
                                num_workers=args.w, 
                                pin_memory=False, drop_last=True)
    
    ##### Train & Test
    print('[*] start a train & test loop')
    best_model_path = os.path.join(args.checkpoint_dir,'best.pth.tar')

    for epoch in range(args.start_epoch, args.epochs):
        train(args, epoch, train_loader, val_loader, model, device, optimizer, writer ,scheduler)
        
        '''
        have to modify best loss
        '''
        #acc = test(args, epoch, test_loader, model, device, writer)
        acc = 0
        
        save_name = '{}.pth.tar'.format(epoch)
        save_name = os.path.join(args.checkpoint_dir, save_name)
        save_model(save_name, epoch, model, optimizer ,scheduler)

        if epoch == 0:
            best_acc = acc
            save_model(best_model_path, epoch, model, optimizer , scheduler)

        else:
            if best_acc < acc:
                best_acc = acc
                save_model(best_model_path, epoch, model, optimizer , scheduler)

    ##### Evaluation (with best model)
    print("[*] Best Model's Results")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['state_dict'])


if __name__ == '__main__':
    argv = parse_arguments(sys.argv[1:])
    main(argv)