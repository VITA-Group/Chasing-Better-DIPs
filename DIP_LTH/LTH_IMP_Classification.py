from __future__ import print_function
from DIP_models import *
from utils.denoising_utils import *
import matplotlib
import matplotlib.pyplot as plt
import warnings
import os
import argparse
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from LTH_utils import *
from pruning_utils import *
import torch
import torchvision
import torchvision.transforms as transforms
from DIP_models import *
from utils.denoising_utils import *
from LTH_utils import *
from pruning_utils import *


matplotlib.use('agg')
warnings.filterwarnings("ignore")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor


def args():
    parser = argparse.ArgumentParser(description='LTH_DIP_model_cifar10')

    # fixme: ------------------------- LTH args settings -------------------------
    # ------------------------------------ basic setting ------------------------------------
    parser.add_argument('--gpu', type=int, default=1, help='gpu device id')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--resume', action="store_true", help="resume from checkpoint")  # FIXME, 触发时为True
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint file')
    parser.add_argument('--print_freq', default=50, type=int, help='print frequency')
    parser.add_argument('--save_dir', help='The directory used to save the trained models',
                        default='./LTH_IMP_classification',
                        type=str)

    # ------------------------------------ training setting ------------------------------------
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=2e-4, type=float, help='weight decay')
    parser.add_argument('--decreasing_lr', default='91,136', help='decreasing strategy')

    # ------------------------------------ Pruning setting ------------------------------------
    parser.add_argument('--pruning_times', default=25, type=int,
                        help='overall times of pruning, we find after 40, the value tends to be the same')
    parser.add_argument('--rate', default=0.2, type=float, help='pruning rate')
    parser.add_argument('--prune_type', default='lt', type=str, help='IMP type (lt, pt, rewind_lt or pt_trans)')
    parser.add_argument('--random_prune', action="store_true", help="whether using random pruning")
    parser.add_argument('--pretrained', default=None, type=str, help='pretrained weight for pt')
    parser.add_argument('--conv1', action="store_true", help="whether pruning & loading conv1")
    parser.add_argument('--fc', action="store_true", help="whether loading fc")
    parser.add_argument('--rewind_epoch', default=9, type=int, help='rewind checkpoint')

    parser.add_argument('--save_dir_name', default='F16', type=str, help='F16 | snail')
    parser.add_argument('--find_best_psnr_model_dir', default='find_best_DIP_model', type=str)

    # load the tickets.
    parser.add_argument('--cifar_train_iters', default=182, type=int)
    parser.add_argument('--ticket_name', default='./tickets/LTH-no-rewind-ticket-p-10-ptimes-4-psnr-30-78.pth.tar',
                        type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = args()
    os.makedirs(args.save_dir, exist_ok=True)

    if args.seed:
        setup_seed(args.seed)

    if args.random_prune:
        print('Random Unstructure Pruning')
    else:
        print('L1 Unstructure Pruning')

    torch.cuda.set_device(int(args.gpu))

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                              shuffle=True, num_workers=1, drop_last=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    input_depth = 3

    from NASDIP_models.skip import skip_classifier

    model = skip_classifier(num_input_channels=input_depth,
                            num_output_channels=3,
                            num_channels_down=[128] * 5,
                            num_channels_up=[128] * 5,
                            num_channels_skip=[4] * 5,
                            upsample_mode='bilinear',
                            downsample_mode='stride',
                            need_sigmoid=True,
                            need_bias=True,
                            pad='reflection',
                            act_fun='LeakyReLU')

    initalization = torch.load('./init_weight_NASDIP.pt', map_location=torch.device('cuda:' + str(args.gpu)))
    model = model.type(dtype)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in initalization.items() if k in model_dict
                       and k != '1.0.1.1.weight' and k != '1.1.1.1.weight'}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.cuda()

    decreasing_lr = list(map(int, args.decreasing_lr.split(',')))

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)

    for state in range(0, args.pruning_times):

        print('******************************************')
        print('pruning state', state)
        print('******************************************')

        sparsity = check_sparsity(model, conv1=args.conv1)

        all_result = {}
        all_result['train_loss'] = []
        all_result['train_acc'] = []

        correct = 0
        total = 0

        for epoch in range(args.cifar_train_iters):

            model.train()
            loss = torch.tensor([0.0]).float().cuda()

            for i, data in enumerate(trainloader):
                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda()

                optimizer.zero_grad()
                outputs = model(F.interpolate(inputs, size=[50, 50]))

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            correct = 0
            total = 0
            model.eval()

            with torch.no_grad():
                for data in testloader:
                    images, labels = data
                    images, labels = images.cuda(), labels.cuda()
                    outputs = model(F.interpolate(images, size=[50, 50]))
                    _, predicted = torch.max(outputs.data, 1)  # 返回每一行中最大值的那个元素，且返回其索引
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            scheduler.step()

            all_result['train_loss'].append(loss.item())
            all_result['train_acc'].append(correct / total)

        f = open(os.path.join(args.save_dir, 'train_acc_prune_state_%d.txt') % state, 'a')
        f.write('%s \n' % str(correct / total))
        f.close()

        plt.plot(all_result['train_acc'], label='train_acc')
        plt.legend()
        plt.savefig(os.path.join(args.save_dir, 'Train_acc_prune_state_%d.png')%state)
        plt.close()

        if args.random_prune:
            pruning_model_random(model, args.rate, conv1=args.conv1)
        else:
            pruning_model(model, args.rate, conv1=args.conv1)

        current_mask = extract_mask(model.state_dict())

        # todo: save the mask
        torch.save(current_mask, os.path.join(
            args.save_dir, 'mask_prune_state_%d.pt') % (state+1))

        remove_prune(model, conv1=args.conv1)

        prune_model_custom(model, current_mask, conv1=args.conv1)

        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)
