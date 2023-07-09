from __future__ import print_function
from models.downsampler import Downsampler
from utils.sr_utils import *
import torch.optim
from utils.inpainting_utils import *
import torch.optim
from skimage.measure import compare_psnr
from NASDIP_utils.inpainting_utils import *
from NASDIP_models import *
from utils.common_utils import *
import matplotlib
import cv2
import numpy as np
import torch.optim
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


def rgb2ycbcr(im_rgb):
    im_rgb = im_rgb.astype(np.float32)
    im_ycrcb = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2YCR_CB)
    im_ycbcr = im_ycrcb[:,:,(0,2,1)].astype(np.float32)
    im_ycbcr[:,:,0] = (im_ycbcr[:,:,0]*(235-16)+16)/255.0 #to [16/255, 235/255]
    im_ycbcr[:,:,1:] = (im_ycbcr[:,:,1:]*(240-16)+16)/255.0 #to [16/255, 240/255]
    return im_ycbcr

def compare_psnr_y(x, y):
    return compare_psnr(rgb2ycbcr(x.transpose(1,2,0))[:,:,0], rgb2ycbcr(y.transpose(1,2,0))[:,:,0])

import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.enabled       = True
torch.backends.cudnn.benchmark     = True
torch.backends.cudnn.deterministic = True
dtype = torch.cuda.FloatTensor


def parse_args():
    parser = argparse.ArgumentParser(description='NAS-DIP Super-resolution')

    parser.add_argument('--optimizer', dest='optimizer',default='adam', type=str)
    parser.add_argument('--num_iter', dest='num_iter', default=2000, type=int)
    parser.add_argument('--factor', dest='factor', default=4, type=int)
    # parser.add_argument('--show_every', dest='show_every', default=100, type=int)
    # parser.add_argument('--lr', dest='lr', default=0.01, type=float)
    parser.add_argument('--plot', dest='plot', default=False, type=bool)
    parser.add_argument('--noise_method', dest='noise_method',default='noise', type=str)
    parser.add_argument('--input_depth', dest='input_depth', default=32, type=int)
    parser.add_argument('--output_path', dest='output_path',default='results/sr', type=str)
    parser.add_argument('--random_seed', dest='random_seed',default=0, type=int)
    parser.add_argument('--net', dest='net',default='default', type=str)
    parser.add_argument('--reg_noise_std', dest='reg_noise_std', default=0.03, type=float)
    parser.add_argument('--i_NAS', dest='i_NAS', default=-1, type=int)
    parser.add_argument('--job_index', dest='job_index', default=1, type=int)
    parser.add_argument('--save_png', dest='save_png', default=0, type=int)

    # fixme: ------------------------- LTH args settings -------------------------
    # ------------------------------------ basic setting ------------------------------------
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--resume', action="store_true", help="resume from checkpoint")  # FIXME, 触发时为True
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint file')
    parser.add_argument('--print_freq', default=50, type=int, help='print frequency')
    parser.add_argument('--save_dir', help='The directory used to save the trained models',
                        default='./Clean_IMP_super-resolution-Factor-8-4',
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
    # parser.add_argument('--reg_noise_std', default=1. / 30., type=float, help='# set to 1./20. for sigma=50')
    parser.add_argument('--LR', default=0.01, type=float, help='the DIP Optimizer learning rate')
    parser.add_argument('--OPTIMIZER', default='adam', type=str, help='adam | sgd | LBFGS')
    # parser.add_argument('--show_every', default='100', type=int, help='the frequency of showing figs')
    parser.add_argument('--exp_weight', default=0.99, type=float)

    parser.add_argument('--num_iter_snail', default=2400, type=int, help='default 2400, 4800 to over-fitting')
    parser.add_argument('--num_iter_F16', default=3000, type=int, help='default 3000, 6000 to over-fitting')
    # this is for the store dir
    # parser.add_argument('--find_best_psnr_model_dir', default='find_best_DIP_model', type=str)
    parser.add_argument('--pretrained', default='./init_weight_NASDIP.pt', type=str)
    parser.add_argument('--mask_dir', default='./main_imp_LTH_DIP_clean_target/F16/mask_prune_state_6.pt', type=str)
    parser.add_argument('--reverse_mask', action="store_true", help="whether using reverse mask")

    args = parser.parse_args()
    return args


def optimize_SR(parameters, closure, LR, num_iter):
    print('Starting optimization with ADAM')
    optimizer = torch.optim.Adam(parameters, lr=LR)
    for j in range(num_iter):
        optimizer.zero_grad()
        closure()
        optimizer.step()


if __name__ == '__main__':

    args = parse_args()

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

    if args.seed:
        setup_seed(args.seed)
    os.makedirs(os.path.join(args.save_dir), exist_ok=True)

    PSNR_mat = np.empty((0, args.num_iter), dtype=np.float32)

    # todo: SR need set5 and 14 all images
    img_path_list = ['butterfly', 'baboon','pepper','zebra']

    factor_list = [4,8]

    for iter in range(1,4):

        # todo: 4x or 8x super resolution image
        for factor in factor_list:

            for image_name in img_path_list:

                # Choose figure
                img_path = 'data/denoising/set_all/' + image_name + '.png'
                imgs = load_LR_HR_imgs_sr(img_path , -1, factor, 'CROP')

                from NASDIP_models.skip import skip

                model = skip(num_input_channels=args.input_depth,
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

                pretrained = './init_weight_NASDIP.pt'
                initalization = torch.load(pretrained, map_location=torch.device('cuda:' + str(args.gpu)))
                model.load_state_dict(initalization)

                model.cuda()
                model = model.type(dtype)

                # todo: the mask_name, 0 is the dense model-DIP
                for state in range(0, args.pruning_times):

                    print('******************************************')
                    print('pruning state', state)
                    print('******************************************')

                    check_sparsity(model, conv1=args.conv1)

                    # z torch.Size([1, 32, tH, tW])
                    net_input = get_noise(args.input_depth, args.noise_method, (imgs['HR_pil'].size[1], imgs['HR_pil'].size[0])).type(dtype).detach()

                    # Loss
                    mse = torch.nn.MSELoss().type(dtype)

                    # x0 torch.Size([1, 3, H, W])
                    img_LR_var      = np_to_torch(imgs['LR_np']).type(dtype)
                    downsampler     = Downsampler(n_planes=3, factor=factor, kernel_type='lanczos2', phase=0.5, preserve_size=True).type(dtype)

                    psnr_gt_best    = 0

                    # Main
                    i  = 0

                    def closure():

                        global i, net_input, psnr_gt_best

                        # Add variation
                        if args.reg_noise_std > 0:
                            net_input = net_input_saved + (noise.normal_() * args.reg_noise_std)

                        out_HR = model(net_input)      # torch.Size([1, 3, tH, tW]): x
                        out_LR = downsampler(out_HR)  # torch.Size([1, 3, H, W])

                        total_loss = mse(out_LR, img_LR_var)
                        total_loss.backward()

                        # psnr_LR = compare_psnr(imgs['LR_np'], torch_to_np(out_LR))
                        # psnr_HR = compare_psnr(imgs['HR_np'], torch_to_np(out_HR))

                        q1 = torch_to_np(out_HR)[:3].sum(0)
                        t1 = np.where(q1.sum(0) > 0)[0]
                        t2 = np.where(q1.sum(1) > 0)[0]

                        psnr_HR = compare_psnr_y(imgs['HR_np'][:3,t2[0] + 4:t2[-1]-4,t1[0] + 4:t1[-1] - 4], \
                                           torch_to_np(out_HR)[:3,t2[0] + 4:t2[-1]-4,t1[0] + 4:t1[-1] - 4])

                        if psnr_HR > psnr_gt_best:
                            psnr_gt_best = psnr_HR

                        # print ('Iteration %05d    Loss %f   PSNR_LR %.3f   PSNR_HR %.3f    Time %.3f'  % (i, total_loss.item(), psnr_LR, psnr_HR, _t['im_detect'].total_time), '\r', end='')
                        print ('Iteration %05d    Loss %f   PSNR_HR %.3f' % (i, total_loss.item(), psnr_HR), '\r', end='')

                        '''
                        # store final img
                        if i == args.num_iter-1:
                            out_HR_np = torch_to_np(out_HR)
                            cv2.imwrite(os.path.join(args.save_dir, 'Factor_%d_img_%s_prune_%d.png')
                                        %(factor,image_name,name),
                                        np.clip(out_HR_np, 0, 1).transpose(1, 2, 0)[:, :, ::-1] * 255)
                        '''
                        i += 1

                        return total_loss

                    net_input_saved = net_input.detach().clone()
                    noise           = net_input.detach().clone()

                    p = get_params('net', model, net_input)
                    optimize_SR(p, closure, args.lr, args.num_iter)

                    f = open(os.path.join(args.save_dir, 'Clean_IMP_SR_Factor_%d_on_img_%s_start_from_dense_iter_%d.txt')
                             % (factor, image_name, iter), 'a')
                    f.write('%s \n' % str(psnr_gt_best))
                    f.close()

                    if args.random_prune:
                        pruning_model_random(model, args.rate, conv1=args.conv1)
                    else:
                        pruning_model(model, args.rate, conv1=args.conv1)

                    current_mask = extract_mask(model.state_dict())

                    # todo: save the mask
                    torch.save(current_mask, os.path.join(
                        args.save_dir, 'mask_prune_state_%d.pt') % (state + 1))

                    remove_prune(model, conv1=args.conv1)
                    model.load_state_dict(initalization)
                    prune_model_custom(model, current_mask, conv1=args.conv1)
