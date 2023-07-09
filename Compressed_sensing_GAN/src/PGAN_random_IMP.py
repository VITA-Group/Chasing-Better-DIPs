from argparse import ArgumentParser
from pytorch_GAN_zoo import hubconf
from PIL import Image
from pytorch_GAN_zoo.models.networks.custom_layers import EqualizedConv2d
import copy
import torch.nn as nn
import torch.nn.utils.prune as prune
from copy import deepcopy
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch
import os
import random


def copy_params(model):
    flatten = deepcopy(list(p.data for p in model.module.parameters()))
    return flatten


class LinearLrDecay(object):
    def __init__(self, optimizer, start_lr, end_lr, decay_start_step, decay_end_step):

        assert start_lr > end_lr
        self.optimizer = optimizer
        self.delta = (start_lr - end_lr) / (decay_end_step - decay_start_step)
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.start_lr = start_lr
        self.end_lr = end_lr

    def step(self, current_step):
        if current_step <= self.decay_start_step:
            lr = self.start_lr
        elif current_step >= self.decay_end_step:
            lr = self.end_lr
        else:
            lr = self.start_lr - self.delta * (current_step - self.decay_start_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        return lr


def compute_gradient_penalty(D, real_samples, fake_samples, phi):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(real_samples.get_device())
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones([real_samples.shape[0], 1], requires_grad=False).to(real_samples.get_device())
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.reshape(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - phi) ** 2).mean()
    return gradient_penalty


def train(args, gen_net: nn.Module, dis_net: nn.Module, gen_optimizer, dis_optimizer,
          gen_avg_param, train_loader, train_global_steps, epoch, schedulers=None):
    np.random.seed(args.random_seed + epoch ** 2)
    gen_step = 0

    # train mode
    gen_net = gen_net.train()
    dis_net = dis_net.train()
    tps = []
    tns = []
    fns = []
    fps = []

    gen_net.train()

    for iter_idx, (imgs, _) in enumerate(train_loader):
        global_steps = train_global_steps + iter_idx

        # Adversarial ground truths
        real_imgs = imgs.type(torch.cuda.FloatTensor)

        # Sample noise as generator input
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim)))

        dis_optimizer.zero_grad()


        real_validity = dis_net(real_imgs)
        fake_imgs = gen_net(z).detach()

        assert fake_imgs.size() == real_imgs.size()

        fake_validity = dis_net(fake_imgs)

        gradient_penalty = compute_gradient_penalty(dis_net, real_imgs, fake_imgs.detach(), args.phi)
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + gradient_penalty * 10 / (args.phi ** 2)
        d_loss += (torch.mean(real_validity) ** 2) * 1e-3

        d_loss.backward()
        dis_optimizer.step()

        tp = torch.sum(real_validity > 0)
        tn = torch.sum(fake_validity < 0)
        fn = torch.sum(real_validity <= 0)
        fp = torch.sum(fake_validity >= 0)

        precision = tp / (tp + fp + 1e-3)
        recall = tp / (tp + fn + 1e-3)
        accuracy = (tp + tn) / (tp + fn + fp + tn)

        fps.append(fp.item())
        tps.append(tp.item())
        fns.append(fn.item())
        tns.append(tn.item())

        # -----------------
        #  Train Generator
        # -----------------

        if global_steps % args.n_critic == 0:
            gen_optimizer.zero_grad()

            gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.gen_batch_size, args.latent_dim)))
            gen_imgs = gen_net(gen_z)

            fake_validity = dis_net(gen_imgs)

            # cal loss
            g_loss = -torch.mean(fake_validity)

            g_loss.backward()
            gen_optimizer.step()

            # adjust learning rate
            if schedulers:
                gen_scheduler, dis_scheduler = schedulers
                g_lr = gen_scheduler.step(global_steps)
                d_lr = dis_scheduler.step(global_steps)

            # moving average weight
            for p, avg_p in zip(gen_net.parameters(), gen_avg_param):
                avg_p.mul_(0.999).add_(0.001, p.data)

            # writer.add_scalar('g_loss', g_loss.item(), global_steps)
            gen_step += 1

        # verbose
        # print(gen_step, iter_idx, args.print_freq)
        if gen_step and iter_idx % args.print_freq == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                (epoch, args.max_epoch, iter_idx % len(train_loader), len(train_loader), d_loss.item(), g_loss.item()))

        # writer_dict['train_global_steps'] = global_steps + 1

    precision_epoch = sum(tps) / (sum(tps) + sum(fps) + 1e-3)
    recall_epoch = sum(tps) / (sum(tps) + sum(fns) + 1e-3)
    accuracy_epoch = (sum(tps) + sum(tns)) / (sum(tps) + sum(tns) + sum(fps) + sum(fns) + 1e-3)

    return precision_epoch, recall_epoch, accuracy_epoch


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
                    # print('name: \n',name)
                    # print('m_1: \n',m_1)
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
                    parameters_to_prune.append((m_1, 'weight'))

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

        if 'state_dict' in current_mask_weight.keys():
            current_mask_weight = current_mask_weight['state_dict']

        current_mask = extract_mask(current_mask_weight)
        print('current_mask.keys():\n', current_mask.keys())

        prune_model_custom(model, current_mask)
        # see_remain_rate(model)


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
    for name, m in model.module.named_modules():
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


torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


def args():
    PARSER = ArgumentParser()
    # Pretrained model
    PARSER.add_argument('--save_dir', type=str, default='PGAN_random_IMP')
    PARSER.add_argument('--device', type=str, default='cuda:0')
    PARSER.add_argument('--N_ITER', type=int, default=1500)
    PARSER.add_argument('--gpu', type=int, default=0)
    PARSER.add_argument('--prune_state', type=int, default=6)
    PARSER.add_argument('--ratio', type=float, default=0.4)
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


if __name__ == '__main__':

    args = args()

    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    os.makedirs(args.save_dir, exist_ok=True)

    device = args.device

    args.image_size = 256
    args.input_path = './CelebA-HQ'
    trans = transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    dataset = datasets.ImageFolder(args.input_path, transform=trans)

    # args.ratio = 0.4
    subset = torch.utils.data.Subset(dataset, np.arange(int(len(dataset) * args.ratio)))
    train_loader = torch.utils.data.DataLoader(subset, batch_size=args.batch_size, drop_last=False, shuffle=True, num_workers=16)

    # args.max_epoch = 20
    start_epoch = 0
    train_global_steps = start_epoch * len(train_loader)

    kwargs = {'model_name': 'celebAHQ-256',
              'useGPU': True}

    model = hubconf.PGAN(pretrained=True, **kwargs)
    model = nn.DataParallel(model)
    # model = hubconf.PGAN(pretrained=False, **kwargs)

    gen_net = model.module.netG.cuda()
    dis_net = model.module.netD.cuda()

    print('pretrained_weight_G:\n', gen_net.state_dict().keys())

    initial_dis_net_weight = deepcopy(dis_net.state_dict())
    initial_gen_net_weight = deepcopy(gen_net.state_dict())

    #
    for state in range(0,args.prune_state+1):
        print("gen_net_sparsity:\n")
        check_sparsity(model.module.netG)
        print("dis_net_sparsity:\n")
        check_sparsity(model.module.netD)

        # todo IMP GAN
        gen_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gen_net.parameters()),
                                         args.g_lr, betas=(0, 0.99))
        dis_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, dis_net.parameters()),
                                         args.d_lr, betas=(0, 0.99))
        gen_avg_param = copy_params(gen_net)

        #
        #
        precision_epoch = 0
        recall_epoch = 0
        accuracy_epoch = 0

        #
        # todo train PGAN
        for epoch in (range(0, int(args.max_epoch))):
            # train
            precision_epoch, recall_epoch, accuracy_epoch = \
                train(args, model.module.netG, model.module.netD, gen_optimizer,
                      dis_optimizer,
                      gen_avg_param, train_loader,
                      train_global_steps, epoch)

            train_global_steps += len(train_loader)

            print("State_{},\t accuracy_epoch: {}\n".format(state, accuracy_epoch))

        #
        f = open(os.path.join(args.save_dir, 'PGAN_final_accuracy_epoch_from_state_0.txt'), 'a')
        f.write('%s \n' % str(accuracy_epoch))
        f.close()

        #
        # save the dense model
        if state == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'Dense_iter_{}_trained.pth'.format(state)))

        #
        # Generator
        pruning_generate(model.module.netG, 0.2, method='random')
        gen_current_mask = extract_mask(model.module.netG.state_dict())

        pruning_generate(model.module.netD, 0.2, method='random')
        dis_current_mask = extract_mask(model.module.netD.state_dict())

        remove_prune(model.module.netG)
        remove_prune(model.module.netD)

        # save the sparse G & D
        torch.save(model.state_dict(),os.path.join(args.save_dir, 'Iter_{}_sparse_trained.pth'.format(state)))

        #
        model.module.netG.load_state_dict(initial_gen_net_weight)
        model.module.netD.load_state_dict(initial_dis_net_weight)

        #
        prune_model_custom(model.module.netG, gen_current_mask)
        prune_model_custom(model.module.netD, dis_current_mask)
