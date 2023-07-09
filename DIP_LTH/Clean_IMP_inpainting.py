from __future__ import print_function
from utils.inpainting_utils import *
import torch.optim
from skimage.measure import compare_psnr
from NASDIP_utils.inpainting_utils import *
from NASDIP_models import *
from utils.common_utils import *
import matplotlib
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
import warnings

warnings.filterwarnings("ignore")
matplotlib.use('agg')
torch.backends.cudnn.enabled       = True
torch.backends.cudnn.benchmark     = True
torch.backends.cudnn.deterministic = True
dtype = torch.cuda.FloatTensor


def parse_args():
    parser = argparse.ArgumentParser(description='NAS-DIP Denoising')

    parser.add_argument('--optimizer', dest='optimizer',default='adam', type=str)
    parser.add_argument('--num_iter', dest='num_iter', default=3000, type=int)
    parser.add_argument('--plot', dest='plot', default=True, type=bool)
    parser.add_argument('--noise_method', dest='noise_method',default='noise', type=str)
    parser.add_argument('--input_depth', dest='input_depth', default=32, type=int)
    parser.add_argument('--output_path', dest='output_path',default='results/restoration', type=str)
    parser.add_argument('--batch_size', dest='batch_size',default=1, type=int)
    parser.add_argument('--random_seed', dest='random_seed',default=0, type=int)
    parser.add_argument('--net', dest='net',default='default', type=str)
    parser.add_argument('--reg_noise_std', dest='reg_noise_std', default=0.03, type=float)
    parser.add_argument('--i_NAS', dest='i_NAS', default=-1, type=int)
    parser.add_argument('--save_png', dest='save_png', default=0, type=int)

    # fixme: ------------------------- LTH args settings -------------------------
    # ------------------------------------ basic setting ------------------------------------
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--resume', action="store_true", help="resume from checkpoint")  # FIXME, 触发时为True
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint file')
    parser.add_argument('--print_freq', default=50, type=int, help='print frequency')
    parser.add_argument('--save_dir', help='The directory used to save the trained models',
                        default='./Clean_IMP_inpainting',
                        type=str)

    # ------------------------------------ training setting ------------------------------------
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=2e-4, type=float, help='weight decay')
    parser.add_argument('--decreasing_lr', default='91,136', help='decreasing strategy')

    # ------------------------------------ Pruning setting ------------------------------------
    parser.add_argument('--pruning_times', default=16, type=int,
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
    parser.add_argument('--show_every', default='100', type=int, help='the frequency of showing figs')
    parser.add_argument('--exp_weight', default=0.99, type=float)

    parser.add_argument('--num_iter_snail', default=2400, type=int, help='default 2400, 4800 to over-fitting')
    parser.add_argument('--num_iter_F16', default=3000, type=int, help='default 3000, 6000 to over-fitting')
    # this is for the store dir
    parser.add_argument('--save_dir_name_kate', default='kate', type=str, help='F16 | snail')
    # parser.add_argument('--find_best_psnr_model_dir', default='find_best_DIP_model', type=str)
    parser.add_argument('--pretrained', default='./init_weight_NASDIP.pt', type=str)
    parser.add_argument('--mask_dir', default='./main_imp_LTH_DIP_clean_target/F16/mask_prune_state_6.pt', type=str)
    parser.add_argument('--reverse_mask', action="store_true", help="whether using reverse mask")

    args = parser.parse_args()
    return args


def optimize_inpainting(parameters, closure, LR, num_iter):
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

    # Choose figure
    img_path_list = ['kate','library','vase']

    for iter in range(1,4):

        for image_name in img_path_list:

            img_path = 'data/inpainting/' + image_name + '.png'

            # Load image
            img_pil, img_np = get_image(img_path, -1)
            img_np          = nn.ReflectionPad2d(1)(np_to_torch(img_np))[0].numpy()
            img_pil         = np_to_pil(img_np)

            img_mask    = get_bernoulli_mask(img_pil, 0.50)
            img_mask_np = pil_to_np(img_mask)

            img_masked  = img_np * img_mask_np
            mask_var    = np_to_torch(img_mask_np).type(dtype)

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

            initalization = torch.load('./init_weight_NASDIP.pt', map_location=torch.device('cuda:' + str(args.gpu)))
            model.load_state_dict(initalization)

            model.cuda()
            model = model.type(dtype)

            for state in range(0, args.pruning_times):
                print('******************************************')
                print('pruning state', state)
                print('******************************************')

                # z torch.Size([1, 32, 512, 512])
                net_input = get_noise(args.input_depth, args.noise_method, img_np.shape[1:]).type(dtype).detach()

                # Loss
                mse = torch.nn.MSELoss().type(dtype)

                # x0
                img_var = np_to_torch(img_np).type(dtype)

                net_input_saved = net_input.detach().clone()
                noise           = net_input.detach().clone()

                last_net         = None
                psrn_masked_last = 0
                psnr_gt_best     = 0

                # Main
                i  = 0

                def closure():

                    global i, psrn_masked_last, last_net, net_input, psnr_gt_best

                    # Add variation
                    if args.reg_noise_std > 0:
                        net_input = net_input_saved + (noise.normal_() * args.reg_noise_std)

                    out = model(net_input)

                    total_loss = mse(out * mask_var, img_var * mask_var)
                    total_loss.backward()

                    psrn_masked = compare_psnr(img_masked, out.detach().cpu().numpy()[0] * img_mask_np)
                    psrn        = compare_psnr(img_np, out.detach().cpu().numpy()[0])

                    psnr_gt_best = max(psrn, psnr_gt_best)

                    print ('Iteration %05d    Loss %f   PSNR_masked %f PSNR %f'
                           % (i, total_loss.item(), psrn_masked, psrn), '\r', end='')

                    # Backtracking
                    if args.plot and i % args.show_every == 0:
                        out_np = torch_to_np(out)

                        if psrn_masked - psrn_masked_last < -5:
                            print('Falling back to previous checkpoint.')

                            for new_param, net_param in zip(last_net, model.parameters()):
                                net_param.data.copy_(new_param.cuda())

                            return total_loss*0
                        else:
                            last_net = [x.cpu() for x in model.parameters()]
                            psrn_masked_last = psrn_masked

                    i += 1

                    return total_loss

                p = get_params('net', model, net_input)
                optimize_inpainting(p, closure,args.lr, args.num_iter)

                f = open(os.path.join(args.save_dir, 'Best_PSNR_Inpainting_on_img_%s_per_prune_from_dense_iter_%d.txt')
                         % (image_name, iter), 'a')
                f.write('%s \n' % str(psnr_gt_best))
                f.close()

                check_sparsity(model, conv1=args.conv1)

                if args.random_prune:
                    pruning_model_random(model, args.rate, conv1=args.conv1)
                else:
                    pruning_model(model, args.rate, conv1=args.conv1)

                current_mask = extract_mask(model.state_dict())

                # todo: save the mask
                torch.save(current_mask, os.path.join(
                    args.save_dir, 'mask_prune_state_%d.pt') % (state+1))

                remove_prune(model, conv1=args.conv1)
                model.load_state_dict(initalization)
                prune_model_custom(model, current_mask, conv1=args.conv1)

