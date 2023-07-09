from argparse import ArgumentParser
from pytorch_GAN_zoo import hubconf
from PIL import Image
import matplotlib.pyplot as plt
from models.networks.custom_layers import EqualizedConv2d
import copy
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch
import os
import random


def extract_mask(model_dict):
    new_dict = {}
    for key in model_dict.keys():
        if 'mask' in key:
            if 'module' in key:
                new_key = key[len('module.'):]
            else:
                new_key = key
            new_dict[new_key] = copy.deepcopy(model_dict[key])
    return new_dict
# todo : special settings for PGAN
def remove_prune(model, conv1=False):
    print('remove pruning')
    for name, m in model.module.named_modules():
        if isinstance(m, EqualizedConv2d):
            for name_1, m_1 in m.named_modules():
                if isinstance(m_1, nn.Conv2d):
                    prune.remove(m_1, 'weight')
# todo : special settings for PGAN
def prune_model_custom(model, mask_dict, conv1=False):
    print('start unstructured pruning with custom mask')

    for name, m in model.named_modules():
        if isinstance(m, EqualizedConv2d):
            for name_1, m_1 in m.named_modules():
                if isinstance(m_1, nn.Conv2d):
                    if name[:17] == 'module.scaleLayer':
                        name = name[7:]  # .0.0
                        prune.CustomFromMask.apply(m_1, 'weight',
                                                   mask=mask_dict[name + '.module.weight_mask'])

                    elif name[:17] == 'module.toRGBLayer':
                        name = name[7:]  # .0
                        prune.CustomFromMask.apply(m_1, 'weight',
                                                   mask=mask_dict[name + '.module.weight_mask'])

                    elif name[:17] == 'module.groupScale':

                        name = name[7:]  # .0
                        prune.CustomFromMask.apply(m_1, 'weight',
                                                   mask=mask_dict[name + '.module.weight_mask'])

# todo : special settings for PGAN
def pruning_generate(model, px, method='l1'):
    parameters_to_prune = []

    # todo: for the PGAN specidal (EqualizedConv2d is a self-defined conv layer)
    for name, m in model.module.named_modules():

        if isinstance(m, EqualizedConv2d):
            for name_1, m_1 in m.named_modules():
                if isinstance(m_1, nn.Conv2d):
                    # print('name:\n',name)
                    # print('m_1:\n',m_1)
                    parameters_to_prune.append((m_1, 'weight'))

    # print("param to prune:\n",parameters_to_prune)

    if method == 'l1':
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=px,
        )

    elif method == 'random':
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.RandomUnstructured,
            amount=px,
        )

def load_ticket_PGAN(model, args, mask_dir):
    if mask_dir:
        current_mask_weight = torch.load(mask_dir, map_location=torch.device('cuda:' + str(args.gpu)))

        # print('mask:\n',current_mask_weight)

        if 'state_dict' in current_mask_weight.keys():
            current_mask_weight = current_mask_weight['state_dict']

        current_mask = extract_mask(current_mask_weight)
        print('current_mask.keys():\n', current_mask.keys())

        prune_model_custom(model, current_mask)
        # see_remain_rate(model)


def data1(n_samples, x, sigma, device):
    n_features = x.contiguous().view(-1).size()[0]
    A = torch.randn([n_samples, n_features], device=device) / np.sqrt(n_samples)
    y = A.mm(x.contiguous().view(-1, 1)) + sigma * torch.randn([n_samples, 1], device=device)
    return y, A


def data2(device, n_outliers, n_features, type_outliers=1, beta=1, rho=1):
    if type_outliers == 1:
        Y = np.ones(n_outliers)
        X = np.ones((n_outliers, n_features))
    elif type_outliers == 2:
        Y = 10000 * torch.ones([n_outliers, 1], device=device)  # 100*torch.ones([n_outliers,1], device=device)
        X = torch.ones([n_outliers, n_features], device=device)
    elif type_outliers == 3:
        Y = np.random.randint(2, size=n_outliers)
        X = np.random.rand(n_outliers, n_features)
    else:
        cov = np.identity(n_features)
        X = feature_mat(n_features, n_outliers, rho)
        Y = X.dot(beta) + sigma * randn(n_samples)
    return Y, X


def data3(device, n_heavy_tail, x, deg=2):
    n_features = x.view(-1).size()[0]
    A = torch.Tensor(np.random.standard_t(2, size=(n_heavy_tail, n_features))).to(device) / np.sqrt(
        n_features)  # (torch.randn([n_heavy_tail, n_features], device=device)**2)/np.sqrt(n_heavy_tail)
    y = A.mm(x.view(-1, 1)) + torch.Tensor(np.random.standard_t(deg, size=(n_heavy_tail, 1))).to(device)
    return y, A


def data_merge(y1, A1, y2, A2):
    y = torch.cat((y1, y2))
    A = torch.cat((A1, A2))
    return y, A


def log_normal_data(n_samples, x, sigma):
    n_features = x.view(-1, 1).size()[0]
    A = torch.randn(n_samples, n_features)
    A = torch.exp(A)
    y = A.mm(x.view(-1, 1)) + sigma * torch.exp(torch.randn(n_samples, 1))
    return y, A


def mom_obj(A, y, zf, zg, batch_size):
    shuffled_idx = torch.randperm(A.size()[0])
    A_shuffled = A.clone()[shuffled_idx, :]
    y_shuffled = y.clone().view(-1)[shuffled_idx].view(y.size())

    # compute AG(zf), AG(zg)
    Af = A_shuffled.mm(gen(zf).view(-1, 1))
    Ag = A_shuffled.mm(gen(zg).view(-1, 1))

    # find (y_i - a_iG(zf))^2 , (y_i - a_iG(zg))^2
    loss_1 = se_unreduced(Af, y_shuffled)
    loss_2 = se_unreduced(Ag, y_shuffled)

    # now find median block of loss_1 - loss_2
    loss_3 = loss_1 - loss_2
    loss_3 = loss_3[:batch_size * (A.shape[0] // batch_size)]  # make the number of rows a multiple of batch size
    loss_3 = loss_3.view(-1, batch_size)  # reshape
    loss_3 = loss_3.mean(axis=1)  # find mean on each batch
    loss_3_numpy = loss_3.detach().cpu().numpy()  # convert to numpy

    median_idx = np.argsort(loss_3_numpy)[loss_3_numpy.shape[0] // 2]  # sort and pick middle element

    # pick median block
    loss_1_mom = loss_1[median_idx * batch_size: batch_size * (median_idx + 1), :]
    loss_2_mom = loss_2[median_idx * batch_size: batch_size * (median_idx + 1), :]
    loss_f = torch.mean(loss_1_mom - loss_2_mom)

    return loss_f


def get_l2_loss(image1, image2):
    """Get L2 loss between the two images"""
    assert image1.shape == image2.shape
    return np.mean((image1 - image2) ** 2)


def scale(image):
    min_val = image.min()
    max_val = image.max()
    return (image - min_val) / (max_val - min_val)
def get_image(path, imsize=-1):
    """Load an image and resize to a cpecific size.

    Args:
        path: path to image
        imsize: tuple or scalar with dimensions; -1 for `no resize`
    """
    img = load(path)

    if isinstance(imsize, int):
        imsize = (imsize, imsize)

    if imsize[0] != -1 and img.size != imsize:
        if imsize[0] > img.size[0]:
            img = img.resize(imsize, Image.BICUBIC)
        else:
            img = img.resize(imsize, Image.ANTIALIAS)

    img_np = pil_to_np(img)

    return img_np
def load(path):
    """Load PIL image."""
    img = Image.open(path)
    return img
def pil_to_np(img_PIL):
    '''Converts image in PIL format to np.array.

    From W x H x C [0...255] to C x W x H [0..1]
    '''
    ar = np.array(img_PIL)

    if len(ar.shape) == 3:
        ar = ar.transpose(2, 0, 1)
    elif len(ar.shape) == 4:
        ar = ar.transpose(0, 3, 1, 2)
    else:
        ar = ar[None, ...]

    return ar.astype(np.float32) / 255.
def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)[None, :]
def torch_to_np(img_var):
    '''Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.detach().cpu().numpy()[0]


def check_sparsity(model):
    sum_list = 0
    zero_sum = 0
    for name, m in model.named_modules():
        if isinstance(m, EqualizedConv2d):


            for name_1, m_1 in m.named_modules():

                if isinstance(m_1, nn.Conv2d):


                    sum_list = sum_list + float(m_1.weight.nelement())
                    zero_sum = zero_sum + float(torch.sum(m_1.weight == 0))

    print('* remain weight = ', 100 * (1 - zero_sum / sum_list), '%')

    return 100 * (1 - zero_sum / sum_list)


def see_remain_rate(model):
    sum_list = 0
    zero_sum = 0

    for m in model.module.modules():
        if isinstance(m, EqualizedConv2d):
            for name_1, m_1 in m.named_modules():
                if isinstance(m_1, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                    sum_list = sum_list + float(m_1.weight.nelement())
                    zero_sum = zero_sum + float(torch.sum(m_1.weight == 0))

    print('remain weight = ', 100 * (1 - zero_sum / sum_list), '%')






def args():
    PARSER = ArgumentParser()
    # Pretrained model
    PARSER.add_argument('--save_dir', type=str, default='PGAN_on_inpaint')
    PARSER.add_argument('--device', type=str, default='cuda:0')
    PARSER.add_argument('--N_ITER', type=int, default=1500)
    PARSER.add_argument('--gpu', type=int, default=0)
    PARSER.add_argument('--prune_state', type=int, default=6)
    PARSER.add_argument('--ratio', type=float, default=0.004)
    PARSER.add_argument('--max_epoch', type=int, default=30)
    PARSER.add_argument('--random_seed', type=int, default=42)
    PARSER.add_argument('--batch_size', type=int, default=64)
    # PARSER.add_argument('--prune_state', type=int, default=16)
    PARSER.add_argument(
        '--latent-dim',
        type=int,
        default=512,
        help='dimensionality of the latent space')
    PARSER.add_argument(
        '--g-lr',
        type=float,
        default=0.0002,
        help='adam: gen learning rate')
    PARSER.add_argument(
        '--d-lr',
        type=float,
        default=0.0002,
        help='adam: disc learning rate')
    PARSER.add_argument(
        '--max-iter',
        type=int,
        default=50000,
        help='set the max iteration number')
    PARSER.add_argument(
        '--n-critic',
        type=int,
        default=1,
        help='number of training steps for discriminator per iter')
    PARSER.add_argument(
        '--phi',
        type=float,
        default=1.0,
        help='phi used in gradient penalty')
    PARSER.add_argument('--lr_decay', default=False)
    PARSER.add_argument(
        '-gen-bs',
        '--gen-batch-size',
        type=int,
        default=1,
        help='size of the batches')
    PARSER.add_argument(
        '-dis-bs',
        '--dis-batch-size',
        type=int,
        default=64,
        help='size of the batches')
    PARSER.add_argument(
        '--print-freq',
        type=int,
        default=10,
        help='interval between each verbose')
    args = PARSER.parse_args()
    return args

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    args = args()

    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    os.makedirs(args.save_dir, exist_ok=True)

    device = args.device

    args.image_size = 256
    args.input_path = './CelebA-HQ-new'
    trans = transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = datasets.ImageFolder(args.input_path, transform=trans)

    args.ratio = 1 # for the new dataset.
    subset = torch.utils.data.Subset(dataset, np.arange(int(len(dataset) * args.ratio)))
    train_loader = torch.utils.data.DataLoader(subset, batch_size=1, drop_last=False, shuffle=False)

    kwargs = {'model_name': 'celebAHQ-256',
              'useGPU': True}
    model = hubconf.PGAN(pretrained=True, **kwargs)
    model = nn.DataParallel(model)

    state = 0

    sparse_weight = torch.load('./trained_weight/random/Iter_0_sparse_trained.pth',
                               map_location=torch.device('cpu'))
    model.load_state_dict(sparse_weight)

    pruning_generate(model.module.netG, 0.2, method='random')
    remove_prune(model.module.netG)

    print('load_sparse_weight_G:\n', model.module.netG.state_dict().keys())

    check_sparsity(model.module.netG)

    mask_list = ['kate']

    for mask_name in mask_list:
        for iter_idx, (imgs, _) in enumerate(train_loader):

            x0 = imgs.type(torch.cuda.FloatTensor)
            x0.to(device)

            model.module.netG.eval()
            model.module.netD.eval()


            def gen(z):
                return model.module.netG(z)

            nz = 512
            n = 3 * 256 * 256
            n_features = n

            img_np = torch_to_np(x0)

            # get the mask
            mask_path = os.path.join('./inpaint_mask/%s_mask.png')%mask_name
            mask_np = get_image(mask_path, imsize=256)

            img_masked = img_np * mask_np

            mask_torch = np_to_torch(mask_np).to(device)
            # corrupted image y
            img_masked_var = np_to_torch(img_masked).to(device)

            z0 = torch.randn(1, 512)
            zf = torch.randn(z0.shape, device=device)
            zf.requires_grad = True

            # learning rate, number of iterations and batch_size
            LRF = 1e-2
            N_ITER = args.N_ITER

            # array for storing error values
            loss_mom = torch.zeros(N_ITER)

            criterion = nn.MSELoss(reduction='none').to(device)

            # adam optimizers
            opt_f = torch.optim.Adam([zf], lr=LRF)

            for i in range(N_ITER):
                opt_f.zero_grad()
                output = gen(zf)*mask_torch

                loss_f = criterion(output,img_masked_var)
                loss_f.backward(loss_f.clone().detach())
                opt_f.step()

                # record error value
                loss_mom[i] = torch.norm(gen(zf).detach() - x0) ** 2 / n_features

                if (i + 1) % 100 == 0:
                    print('i: %d loss: %f' % (i, loss_mom[i].item()))

            MSE_value = get_l2_loss(x0.detach().cpu().numpy(), gen(zf).detach().cpu().numpy())

            # print('MSE_value of two images:\n', MSE_value)

            f = open(os.path.join(args.save_dir, 'MSE_PGAN_prune_state_%d.txt')
                     %state, 'a')

            f.write('%s \n' % str(MSE_value))
            f.close()

            x0_save = scale(x0[0].detach().cpu().numpy())
            gen_save = scale(gen(zf)[0].detach().cpu().numpy())
            img_masked_save = scale(img_masked_var[0].detach().cpu().numpy())

            # plt.imsave(os.path.join(args.save_dir,'GT_prune_%d.png')%state, x0_save.transpose(1, 2, 0))
            plt.imsave(os.path.join(args.save_dir, 'Img_%d_Pred_prune_state_%d.png')
                       % (iter_idx, state), gen_save.transpose(1, 2, 0))

            plt.imsave(os.path.join(args.save_dir, 'Img_%d_masked_img.png')
                       % iter_idx, img_masked_save.transpose(1, 2, 0))
