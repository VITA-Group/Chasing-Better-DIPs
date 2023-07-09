import os
import time
import copy  
import torch
import random
import shutil
import numpy as np  
import torch.nn as nn 
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
# from dataset import *
# from models.resnet import resnet18, resnet50, resnet152
from pruning_utils import *

__all__ = ['setup_model_dataset', 'setup_seed',
            'train', 'test', 
            'save_checkpoint', 'load_weight_pt_trans', 'load_ticket']

def setup_model_dataset(args):
    
    #prepare dataset
    if args.dataset == 'cifar10':
        classes = 10
        train_loader, val_loader, test_loader = cifar10_dataloaders(batch_size= args.batch_size, data_dir =args.data)
    elif args.dataset == 'cifar100':
        classes = 100
        train_loader, val_loader, test_loader = cifar100_dataloaders(batch_size= args.batch_size, data_dir =args.data)
    elif args.dataset == 'svhn':
        classes = 10
        train_loader, val_loader, test_loader = svhn_dataloaders(batch_size= args.batch_size, data_dir =args.data)
    elif args.dataset == 'fmnist':
        classes = 10
        train_loader, val_loader, test_loader = fashionmnist_dataloaders(batch_size= args.batch_size, data_dir =args.data)
    else:
        raise ValueError("Unknown Dataset")

    # prepare model
    if args.arch == 'resnet18':
        model = resnet18(num_classes = classes)
    elif args.arch == 'resnet50':
        model = resnet50(num_classes = classes)
    elif args.arch == 'resnet152':
        model = resnet152(num_classes = classes)
    else:
        raise ValueError("Unknown Model")
    
    if args.dataset == 'fmnist':
        model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)

    return model, train_loader, val_loader, test_loader


# FIXME this is the clean training.
def train(train_loader, model, criterion, optimizer, epoch, args):

    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()
    for i, (image, target) in enumerate(train_loader):

        if epoch < args.warmup:
            warmup_lr(epoch, i+1, optimizer, one_epoch_step=len(train_loader), args=args)

        image = image.cuda()
        target = target.cuda()

        # compute output
        output_clean = model(image)
        loss = criterion(output_clean, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output_clean.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]

        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if i % args.print_freq == 0:
            end = time.time()
            print('Epoch: [{0}][{1}/{2}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                'Time {3:.2f}'.format(
                    epoch, i, len(train_loader), end-start, loss=losses, top1=top1))
            start = time.time()

    print('train_accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg


# FIXME clean test
def test(val_loader, model, criterion, args):
    """
    Run evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (image, target) in enumerate(val_loader):
        
        image = image.cuda()
        target = target.cuda()

        # compute output
        with torch.no_grad():
            output = model(image)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(val_loader), loss=losses, top1=top1))

    print('valid_accuracy {top1.avg:.3f}'
        .format(top1=top1))

    return top1.avg


def save_checkpoint_one_shot_prune(state, is_SA_best, save_path, pruning, filename='checkpoint.pth.tar'):
    # filepath = os.path.join(save_path, 'DIP_epoch_'+str(pruning)+'_model.pth.tar')
    # torch.save(state, os.path.join(save_path, 'DIP_epoch_'+str(pruning)+'_model.pth.tar'))
    torch.save(state, os.path.join(save_path, 'One_shot_rate_'+str(state)+'_model.pth.tar'))


def save_checkpoint(state, is_SA_best, save_path, pruning, filename='checkpoint.pth.tar'):
    # filepath = os.path.join(save_path, 'DIP_epoch_'+str(pruning)+'_model.pth.tar')
    # torch.save(state, os.path.join(save_path, 'DIP_epoch_'+str(pruning)+'_model.pth.tar'))
    torch.save(state, os.path.join(save_path, 'DIP_best_model.pth.tar'))



def load_weight_pt_trans(model, initalization, args): 
    print('loading pretrained weight')
    loading_weight = extract_main_weight(initalization, fc=args.fc, conv1=args.conv1)
    
    for key in loading_weight.keys():
        if not (key in model.state_dict().keys()):
            print(key)
            assert False

    print('*number of loading weight={}'.format(len(loading_weight.keys())))
    print('*number of model weight={}'.format(len(model.state_dict().keys())))
    model.load_state_dict(loading_weight, strict=False)


def load_ticket(model, args, pretrained, mask_dir):

    # weight
    if pretrained:
        initalization = torch.load(pretrained, map_location=torch.device('cuda:' + str(args.gpu)))
        model.load_state_dict(initalization)

    # mask 
    if mask_dir:

        current_mask_weight = torch.load(mask_dir, map_location = torch.device('cuda:'+str(args.gpu)))

        if 'state_dict' in current_mask_weight.keys():
            current_mask_weight = current_mask_weight['state_dict']

        current_mask = extract_mask(current_mask_weight)

        if args.reverse_mask:
            current_mask = reverse_mask(current_mask)
        prune_model_custom(model, current_mask, conv1=args.conv1)

        # check_sparsity(model, conv1=args.conv1)


def load_ticket_cifar(model, args, pretrained, mask_dir):

    # weight
    if pretrained:

        # 第一层不加载参数
        initalization = torch.load(pretrained, map_location=torch.device('cuda:' + str(args.gpu)))
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in initalization.items() if k in model_dict
                           and k != '1.0.1.1.weight' and k != '1.1.1.1.weight'}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    # mask
    if mask_dir:

        current_mask_weight = torch.load(mask_dir, map_location = torch.device('cuda:'+str(args.gpu)))

        if 'state_dict' in current_mask_weight.keys():
            current_mask_weight = current_mask_weight['state_dict']

        current_mask = extract_mask(current_mask_weight)

        if args.reverse_mask:
            current_mask = reverse_mask(current_mask)
        prune_model_custom(model, current_mask, conv1=args.conv1)

        check_sparsity(model, conv1=args.conv1)


def warmup_lr(epoch, step, optimizer, one_epoch_step, args):

    overall_steps = args.warmup*one_epoch_step
    current_steps = epoch*one_epoch_step + step 

    lr = args.lr * current_steps/overall_steps
    lr = min(lr, args.lr)

    for p in optimizer.param_groups:
        p['lr']=lr

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def setup_seed(seed): 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed) 
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True 













