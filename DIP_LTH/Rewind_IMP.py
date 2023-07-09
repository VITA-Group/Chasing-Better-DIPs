from __future__ import print_function
from skimage.measure import compare_psnr
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


def optimize_best_model_no_scheduler(optimizer_type, parameters, closure, LR, num_iter):
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=LR)
        for j in range(num_iter):
            optimizer.zero_grad()
            closure()
            optimizer.step()
    else:
        assert False


def args():
    parser = argparse.ArgumentParser(description='DIP-Settings')

    # fixme: ------------------------- LTH args settings -------------------------
    # ------------------------------------ basic setting ------------------------------------
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--resume', action="store_true", help="resume from checkpoint")  # FIXME, 触发时为True
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint file')
    parser.add_argument('--print_freq', default=50, type=int, help='print frequency')
    parser.add_argument('--save_dir', help='The directory used to save the trained models',
                        default='./rewind_5_10_20',
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


# rewind
if __name__ == '__main__':

    args = args()
    torch.cuda.set_device(int(args.gpu))
    if args.seed:
        setup_seed(args.seed)
    os.makedirs(os.path.join(args.save_dir), exist_ok=True)
    sigma_ = args.sigma / 255.

    img_path_list = ['baby']
    rewind_list = ['5', '10', '20']

    for iter in range(1,4):

        for rewind in rewind_list:

            rewind_weight = torch.load(os.path.join(
                './rewind_weight',
                'DIP_model_rewind_%s.pth')%rewind,
                                       map_location=torch.device('cuda:' + str(args.gpu)))

            args.pruning_times = 16

            for img_name in img_path_list:

                img_path = 'data/denoising/Set5/' + img_name + '.png'
                img_pil = crop_image(get_image(img_path, args.imsize)[0], d=32)
                img_np = pil_to_np(img_pil)
                img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma_)

                num_iter = args.num_iter_F16
                input_depth = 32
                figsize = 4

                from NASDIP_models.skip import skip

                net = skip(num_input_channels=input_depth,
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

                initalization = torch.load(os.path.join('./init_weight_NASDIP%d.pth')%iter,
                                           map_location=torch.device('cuda:' + str(args.gpu)))

                net.load_state_dict(initalization)
                net.cuda()
                model = net
                model = model.type(dtype)

                for state in range(0, args.pruning_times):

                    print('******************************************')
                    print('pruning state', state)
                    print('******************************************')

                    check_sparsity(model, conv1=args.conv1)

                    net_input = get_noise(input_depth, args.INPUT, (img_pil.size[1], img_pil.size[0])).type(dtype).detach()
                    mse = torch.nn.MSELoss().type(dtype)

                    img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)
                    img_torch = np_to_torch(img_np).type(dtype)

                    net_input_saved = net_input.detach().clone()
                    noise = net_input.detach().clone()
                    out_avg = None
                    last_net = None
                    psrn_noisy_last = 0
                    i = 0
                    psrn_gt = 0.00
                    psrn_noisy = 0.00

                    best_psnr_gt = 0
                    best_psnr_noisy = 0

                    def closure():
                        global i, out_avg, psrn_noisy_last, last_net, net_input, best_psnr_gt, best_psnr_noisy, psrn_gt

                        if args.reg_noise_std > 0:
                            net_input = net_input_saved + (noise.normal_() * args.reg_noise_std)
                        out = model(net_input)

                        # Smoothing
                        if out_avg is None:
                            out_avg = out.detach()
                        else:
                            out_avg = out_avg * args.exp_weight + out.detach() * (1 - args.exp_weight)

                        # total_loss = mse(out, img_noisy_torch)
                        total_loss = mse(out, img_torch)
                        total_loss.backward()

                        psrn_noisy = compare_psnr(img_noisy_np, out.detach().cpu().numpy()[0])
                        psrn_gt = compare_psnr(img_np, out_avg.detach().cpu().numpy()[0])


                        best_psnr_gt = max(psrn_gt, best_psnr_gt)
                        best_psnr_noisy = psrn_noisy

                        # Backtracking
                        if i % args.show_every:
                            if psrn_noisy - psrn_noisy_last < -5:
                                print('Falling back to previous checkpoint.')

                                for new_param, net_param in zip(last_net, net.parameters()):
                                    net_param.data.copy_(new_param.cuda())

                                return total_loss * 0
                            else:
                                last_net = [x.detach().cpu() for x in net.parameters()]
                                psrn_noisy_last = psrn_noisy

                        i += 1


                    p = get_params(args.OPT_OVER, net, net_input)

                    optimize_best_model_no_scheduler(args.OPTIMIZER, p, closure, args.LR, num_iter)

                    f = open(os.path.join(args.save_dir, 'Iter%d_Rewind_%s_best_PSNR_img_%s.txt')%(iter,rewind,img_name), 'a')
                    f.write('%s \n' % str(best_psnr_gt))
                    f.close()

                    if args.random_prune:
                        pruning_model_random(model, args.rate, conv1=args.conv1)
                    else:
                        pruning_model(model, args.rate, conv1=args.conv1)

                    current_mask = extract_mask(model.state_dict())

                    torch.save(current_mask, os.path.join(
                        args.save_dir, 'Rewind_%s_mask_prune_state_%d.pt') % (rewind, state + 1))

                    remove_prune(model, conv1=args.conv1)
                    model.load_state_dict(rewind_weight)
                    prune_model_custom(model, current_mask, conv1=args.conv1)
