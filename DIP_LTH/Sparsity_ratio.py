from __future__ import print_function
from NASDIP_utils.denoising_utils import *
from NASDIP_models import *
import matplotlib
import torch.optim
import warnings
import os
import argparse
import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn as cudnn
from LTH_utils import *
from pruning_utils import *

matplotlib.use('agg')
warnings.filterwarnings("ignore")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor


def args():
    parser = argparse.ArgumentParser(description='DIP-Settings')

    # fixme: ------------------------- LTH args settings -------------------------
    # ------------------------------------ basic setting ------------------------------------
    parser.add_argument('--gpu', type=int, default=1, help='gpu device id')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--resume', action="store_true", help="resume from checkpoint")  # FIXME, 触发时为True
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint file')
    parser.add_argument('--print_freq', default=50, type=int, help='print frequency')
    parser.add_argument('--save_dir', help='The directory used to save the trained models',
                        default='./LAYER-WISE-Sparsity-Ratio-Multi_Mask',
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
    # parser.add_argument('--pretrained', default=None, type=str, help='pretrained weight for pt')
    parser.add_argument('--conv1', action="store_true", help="whether pruning & loading conv1")
    parser.add_argument('--fc', action="store_true", help="whether loading fc")
    # parser.add_argument('--rewind_epoch', default=9, type=int, help='rewind checkpoint')

    # fixme -------------------- the DEEP IMAGE PRIOR settings ---------------------------------

    parser.add_argument('--imsize', type=int, default=-1, help='imsize default -1')
    parser.add_argument('--PLOT', default=True, help='True means to plot the fig')
    parser.add_argument("--sigma", default=25, type=int)
    parser.add_argument("--i_first", default='pristine', type=str, help='plot the pristine fig')
    parser.add_argument("--i_second", default='noise_img', type=str, help="plot the noise img")
    parser.add_argument("--fname", default='./data/denoising/Set5/baby.png',
                        help='F16: data/denoising/F16.png | Snail: data/denoising/snail.jpg')
    parser.add_argument("--INPUT", default='noise', type=str, help='noise | meshgrid')
    parser.add_argument('--pad', default='reflection', type=str)
    parser.add_argument('--OPT_OVER', default='net', type=str, help='net | input')
    parser.add_argument('--reg_noise_std', default=1. / 30., type=float, help='# set to 1./20. for sigma=50')
    parser.add_argument('--LR', default=0.01, type=float, help='the DIP Optimizer learning rate')
    parser.add_argument('--OPTIMIZER', default='adam', type=str, help='adam | sgd | LBFGS')
    parser.add_argument('--show_every', default='100', type=int, help='the frequency of showing figs')
    parser.add_argument('--exp_weight', default=0.99, type=float)

    parser.add_argument('--num_iter_snail', default=2400, type=int, help='default 2400, 4800 to over-fitting')
    parser.add_argument('--num_iter_F16', default=3000, type=int, help='default 3000, 6000 to over-fitting')
    # this is for the store dir
    parser.add_argument('--save_dir_name', default='F16', type=str, help='F16 | snail')
    # parser.add_argument('--find_best_psnr_model_dir', default='find_best_DIP_model', type=str)
    parser.add_argument('--pretrained', default='./init_weight_NASDIP.pt', type=str)
    parser.add_argument('--mask_dir', default='./main_imp_LTH_DIP_clean_target/F16/mask_prune_state_6.pt', type=str)
    parser.add_argument('--reverse_mask', action="store_true", help="whether using reverse mask")

    args = parser.parse_args()
    return args


# apply the mask on the model.
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

        check_sparsity(model, conv1=args.conv1)


if __name__ == '__main__':
    args = args()
    torch.cuda.set_device(int(args.gpu))
    if args.seed:
        setup_seed(args.seed)

    input_depth = 32

    os.makedirs(os.path.join(args.save_dir), exist_ok=True)

    for name in range(1, 6):

        from NASDIP_models.skip import skip
        model = skip(num_input_channels=input_depth,
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

        model.cuda()
        model = model.type(dtype)
        pretrained = './init_weight_NASDIP.pt'
        mask_dir = os.path.join('./mask/multi_img_clean_mask','mask_prune_state_%d.pt') % name
        load_ticket(model, args, pretrained, mask_dir)

        sum_list = 0
        zero_sum = 0

        for name_layer, m in model.named_modules():
            if isinstance(m, nn.Conv2d):  # and name_layer != '1.0.1.1' and name_layer != '1.1.1.1':
                sum_list = sum_list + float(m.weight.nelement())
                zero_sum = zero_sum + float(torch.sum(m.weight == 0))

                f = open(os.path.join(args.save_dir,
                                      'Multi_LTH_Sparsity_%d.txt') % name, 'a')
                f.write('%s \n' % str(zero_sum / sum_list))
                f.close()
