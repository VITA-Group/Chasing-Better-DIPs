from __future__ import print_function
from DIP_models import *
from skimage.measure import compare_psnr
from utils.denoising_utils import *
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


def get_image_multi(path, imsize=-1):
    """Load an image and resize to a cpecific size.

    Args:
        path: path to image
        imsize: tuple or scalar with dimensions; -1 for `no resize`
    """
    img = load(path)
    img = img.resize((228, 228))

    if isinstance(imsize, int):
        imsize = (imsize, imsize)

    if imsize[0]!= -1 and img.size != imsize:
        if imsize[0] > img.size[0]:
            img = img.resize(imsize, Image.BICUBIC)
        else:
            img = img.resize(imsize, Image.ANTIALIAS)

    img_np = pil_to_np(img)

    return img, img_np


def process_img(img_path):
    img_pil = crop_image(get_image_multi(img_path, args.imsize)[0], d=32)
    img_np = pil_to_np(img_pil)
    img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma_)
    img_torch = np_to_torch(img_np).type(dtype)
    return img_noisy_pil, img_noisy_np, img_pil, img_np, img_torch


def optimize_best_model_no_scheduler(optimizer_type, parameters, closure, LR, num_iter):

    if optimizer_type == 'adam':
        # print('Starting optimization with ADAM')
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
    parser.add_argument('--gpu', type=int, default=1, help='gpu device id')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--resume', action="store_true", help="resume from checkpoint")  # FIXME, 触发时为True
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint file')
    parser.add_argument('--print_freq', default=50, type=int, help='print frequency')
    parser.add_argument('--save_dir', help='The directory used to save the trained models',
                        default='./LTH-IMP_Multi-Img',
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
    # parser.add_argument('--rewind_epoch', default=9, type=int, help='rewind checkpoint')

    # fixme -------------------- the DEEP IMAGE PRIOR settings ---------------------------------

    parser.add_argument('--imsize', type=int, default=-1, help='imsize default -1')
    parser.add_argument('--PLOT', default=True, help='True means to plot the fig')
    parser.add_argument("--sigma", default=25, type=int)
    parser.add_argument("--i_first", default='pristine', type=str, help='plot the pristine fig')
    parser.add_argument("--i_second", default='noise_img', type=str, help="plot the noise img")
    parser.add_argument("--fname", default='data/denoising/F16_GT.png',
                        help='F16: data/denoising/F16_GT.png | Snail: data/denoising/snail.jpg')
    parser.add_argument("--INPUT", default='noise', type=str, help='noise | meshgrid')
    parser.add_argument('--pad', default='reflection', type=str)
    parser.add_argument('--OPT_OVER', default='net', type=str, help='net | input')
    parser.add_argument('--reg_noise_std', default=1. / 30., type=float, help='# set to 1./20. for sigma=50')
    parser.add_argument('--LR', default=0.01, type=float, help='the DIP Optimizer learning rate')
    parser.add_argument('--OPTIMIZER', default='adam', type=str, help='adam | sgd | LBFGS')
    parser.add_argument('--show_every', default='500', type=int, help='the frequency of showing figs')
    parser.add_argument('--exp_weight', default=0.99, type=float)

    parser.add_argument('--num_iter_snail', default=2400, type=int, help='default 2400, 4800 to over-fitting')
    parser.add_argument('--num_iter_F16', default=3000, type=int, help='default 3000, 6000 to over-fitting')
    # this is for the store dir
    parser.add_argument('--save_dir_name', default='F16', type=str, help='F16 | snail')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = args()

    torch.cuda.set_device(int(args.gpu))
    if args.seed:
        setup_seed(args.seed)
    os.makedirs(os.path.join(args.save_dir), exist_ok=True)
    sigma_ = args.sigma / 255.

    img_path_1 = './data/denoising/face/1.jpg'
    img_path_2 = './data/denoising/face/2.jpg'
    img_path_3 = './data/denoising/face/3.jpg'
    img_path_4 = './data/denoising/face/4.jpg'
    img_path_5 = './data/denoising/face/5.jpg'

    for iter in range(1, 4):

        img_noisy_pil_1, img_noisy_np_1, img_pil_1, img_np_1, img_torch_1 = process_img(img_path_1)
        img_noisy_pil_2, img_noisy_np_2, img_pil_2, img_np_2, img_torch_2 = process_img(img_path_2)
        img_noisy_pil_3, img_noisy_np_3, img_pil_3, img_np_3, img_torch_3 = process_img(img_path_3)
        img_noisy_pil_4, img_noisy_np_4, img_pil_4, img_np_4, img_torch_4 = process_img(img_path_4)
        img_noisy_pil_5, img_noisy_np_5, img_pil_5, img_np_5, img_torch_5 = process_img(img_path_5)

        num_iter = args.num_iter_F16
        input_depth = 32
        figsize = 4

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

        initalization = torch.load(os.path.join('./init_weight_NASDIP%d.pth')
                                   %iter, map_location=torch.device('cuda:' + str(args.gpu)))

        model.load_state_dict(initalization)
        model.cuda()
        model = model.type(dtype)

        for state in range(0, args.pruning_times):

            print('******************************************')
            print('pruning state', state)
            print('******************************************')

            # *******************************************************************************************
            # multi img settings
            # todo: net_input_1 是图片size的噪声
            net_input_1 = get_noise(input_depth, args.INPUT, (img_pil_1.size[1], img_pil_1.size[0])).type(dtype).detach()
            net_input_saved_1 = net_input_1.detach().clone()
            noise_1 = net_input_1.detach().clone()
            img_noisy_torch_1 = np_to_torch(img_noisy_np_1).type(dtype)

            net_input_2 = get_noise(input_depth, args.INPUT, (img_pil_2.size[1], img_pil_2.size[0])).type(dtype).detach()
            net_input_saved_2 = net_input_2.detach().clone()
            noise_2 = net_input_2.detach().clone()
            img_noisy_torch_2 = np_to_torch(img_noisy_np_2).type(dtype)

            net_input_3 = get_noise(input_depth, args.INPUT, (img_pil_3.size[1], img_pil_3.size[0])).type(dtype).detach()
            net_input_saved_3 = net_input_3.detach().clone()
            noise_3 = net_input_3.detach().clone()
            img_noisy_torch_3 = np_to_torch(img_noisy_np_3).type(dtype)

            net_input_4 = get_noise(input_depth, args.INPUT, (img_pil_4.size[1], img_pil_4.size[0])).type(dtype).detach()
            net_input_saved_4 = net_input_4.detach().clone()
            noise_4 = net_input_4.detach().clone()
            img_noisy_torch_4 = np_to_torch(img_noisy_np_4).type(dtype)

            net_input_5 = get_noise(input_depth, args.INPUT, (img_pil_5.size[1], img_pil_5.size[0])).type(dtype).detach()
            net_input_saved_5 = net_input_5.detach().clone()
            noise_5 = net_input_5.detach().clone()
            img_noisy_torch_5 = np_to_torch(img_noisy_np_5).type(dtype)

            # *******************************************************************************************

            mse = torch.nn.MSELoss().type(dtype)
            out_avg = None
            last_net = None
            psrn_noisy_last = 0
            i = 0
            psrn_gt = 0.00
            psrn_noisy = 0.00

            best_PSNR_GT_1 = 0.0
            best_PSNR_GT_2 = 0.0
            best_PSNR_GT_3 = 0.0
            best_PSNR_GT_4 = 0.0
            best_PSNR_GT_5 = 0.0

            def closure():
                global i, out_avg, psrn_noisy_last, last_net, net_input_1, \
                    net_input_2, net_input_3, net_input_4, net_input_5, \
                    psrn_gt, best_PSNR_GT_1, best_PSNR_GT_2, \
                    best_PSNR_GT_3, best_PSNR_GT_4, best_PSNR_GT_5

                if args.reg_noise_std > 0:
                    net_input_1 = net_input_saved_1 + (noise_1.normal_() * args.reg_noise_std)

                out_1 = model(net_input_1)

                if out_avg is None:
                    out_avg = out_1.detach()
                else:
                    out_avg = out_avg * args.exp_weight + out_1.detach() * (1 - args.exp_weight)

                # todo: keep the same noise of 5 images.
                total_loss = mse(out_1, img_torch_1) + mse(out_1, img_torch_2) + mse(out_1, img_torch_3) \
                             + mse(out_1, img_torch_4) + mse(out_1, img_torch_5)

                total_loss.backward()

                psrn_gt_1 = compare_psnr(img_np_1, out_avg.detach().cpu().numpy()[0])
                psrn_gt_2 = compare_psnr(img_np_2, out_avg.detach().cpu().numpy()[0])
                psrn_gt_3 = compare_psnr(img_np_3, out_avg.detach().cpu().numpy()[0])
                psrn_gt_4 = compare_psnr(img_np_4, out_avg.detach().cpu().numpy()[0])
                psrn_gt_5 = compare_psnr(img_np_5, out_avg.detach().cpu().numpy()[0])

                best_PSNR_GT_1 = max(psrn_gt_1, best_PSNR_GT_1)
                best_PSNR_GT_2 = max(psrn_gt_2, best_PSNR_GT_2)
                best_PSNR_GT_3 = max(psrn_gt_3, best_PSNR_GT_3)
                best_PSNR_GT_4 = max(psrn_gt_4, best_PSNR_GT_4)
                best_PSNR_GT_5 = max(psrn_gt_5, best_PSNR_GT_5)

                # Backtracking
                if i % args.show_every:
                    if psrn_noisy - psrn_noisy_last < -5:
                        print('Falling back to previous checkpoint.')

                        for new_param, net_param in zip(last_net, net.parameters()):
                            net_param.data.copy_(new_param.cuda())

                        return total_loss * 0
                    else:
                        last_net = [x.detach().cpu() for x in model.parameters()]
                        psrn_noisy_last = psrn_noisy

                i += 1

            p = get_params(args.OPT_OVER, model, net_input=None)
            optimize_best_model_no_scheduler(args.OPTIMIZER, p, closure, args.LR, num_iter)

            if args.random_prune:
                pruning_model_random(model, args.rate, conv1=args.conv1)
            else:
                pruning_model(model, args.rate, conv1=args.conv1)

            current_mask = extract_mask(model.state_dict())

            # todo: save the mask
            torch.save(current_mask, os.path.join(
                args.save_dir, 'mask_prune_state_%d.pt') % (state+1))

            remove_prune(model)
            model.load_state_dict(initalization)
            prune_model_custom(model, current_mask)
