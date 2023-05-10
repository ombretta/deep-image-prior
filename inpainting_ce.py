'''
Code for "Inpainting" figures $6$, $8$ and 7 (top) from the main paper.
'''


from __future__ import print_function

from models.resnet import ResNet
from models.unet import UNet
from models.skip import skip
from models.ce_skip import ce_skip

from utils.inpainting_utils import *

import os
import argparse

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor

i = 0

def load_image_and_mask(img_path, mask_path, imsize, dim_div_by):
    # Load mask
    img_pil, img_np = get_image(img_path, imsize)
    img_mask_pil, img_mask_np = get_image(mask_path, imsize)

    # Center crop
    img_mask_pil = crop_image(img_mask_pil, dim_div_by)
    img_pil = crop_image(img_pil, dim_div_by)

    img_np = pil_to_np(img_pil)
    img_mask_np = pil_to_np(img_mask_pil)

    # Visualize
    img_mask_var = np_to_torch(img_mask_np).type(dtype)
    plot_image_grid([img_np, img_mask_np, img_mask_np * img_np], 3, 11);

    return img_np, img_mask_np


class InpaintingConfigs(object):

    def __init__(self, net_type, img_name):
        (net, input_depth, input, img_np, img_mask_np, param_noise,
         reg_noise_std, figsize, out_path, OPT_OVER, OPTIMIZER, LR,
         num_iter) = self.get_configs(net_type, img_name)
        self.net = net
        self.input_depth = input_depth
        self.input = input
        self.img_np = img_np
        self.img_mask_np = img_mask_np
        self.param_noise = param_noise
        self.reg_noise_std = reg_noise_std
        self.figsize = figsize
        self.out_path = out_path
        self.OPT_OVER = OPT_OVER
        self.OPTIMIZER = OPTIMIZER
        self.LR = LR
        self.num_iter = num_iter

    
    def get_configs(self, net_type, img_name):
        
        # Input image params
        imsize = -1
        dim_div_by = 64
        img_path = 'data/inpainting/' + img_name + '.png'
        mask_path = 'data/inpainting/' + img_name + '_mask.png'

        # Load image and mask
        img_np, img_mask_np = load_image_and_mask(img_path, mask_path, imsize, dim_div_by)

        # Output paths
        out_path = 'results/inpainting/' + net_type + '/' + img_name + '/'
        if not os.path.exists('results/inpainting/' + net_type):
            os.mkdir('results/inpainting/' + net_type)
        if not os.path.exists(out_path):
            os.mkdir(out_path)

        # Network/optimizer setup
        pad = 'zero'  # 'reflection'
        OPT_OVER = 'net'
        OPTIMIZER = 'adam'

        # Optimization and model
        if 'vase.png' in img_path:
            input = 'meshgrid'
            input_depth = 2
            LR = 0.01
            num_iter = 5001
            param_noise = False
            figsize = 5
            reg_noise_std = 0.03

            if "ce_skip" in net_type:
                net = ce_skip(in_rotations=1, out_rotations=3,
                              num_input_channels=input_depth,
                              num_output_channels=img_np.shape[0],
                              num_channels_down=[128] * 5,
                              num_channels_up=[128] * 5,
                              num_channels_skip=[0] * 5,
                              upsample_mode='nearest', filter_skip_size=1, filter_size_up=3, filter_size_down=3,
                              need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)
            else:
                net = skip(input_depth, img_np.shape[0],
                           num_channels_down=[128] * 5,
                           num_channels_up=[128] * 5,
                           num_channels_skip=[0] * 5,
                           upsample_mode='nearest', filter_skip_size=1, filter_size_up=3, filter_size_down=3,
                           need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)

        elif ('kate.png' in self.img_path) or ('peppers.png' in self.img_path):
            # Same params and net as in super-resolution and denoising
            input = 'noise'
            input_depth = 32
            LR = 0.01
            num_iter = 6001
            param_noise = False
            figsize = 5
            reg_noise_std = 0.03

            if "ce_skip" in net_type:
                net = ce_skip(in_rotations=1, out_rotations=3,
                          num_input_channels=input_depth,
                          num_output_channels=img_np.shape[0],
                          num_channels_down=[128] * 5,
                          num_channels_up=[128] * 5,
                          num_channels_skip=[128] * 5,
                          filter_size_up=3, filter_size_down=3,
                          upsample_mode='nearest', filter_skip_size=1,
                          need_sigmoid=True, need_bias=True, pad=pad,
                          act_fun='LeakyReLU').type(dtype)
            else:
                net = skip(input_depth, img_np.shape[0],
                           num_channels_down=[128] * 5,
                           num_channels_up=[128] * 5,
                           num_channels_skip=[128] * 5,
                           filter_size_up=3, filter_size_down=3,
                           upsample_mode='nearest', filter_skip_size=1,
                           need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)

        elif 'library.png' in self.img_path:

            input = 'noise'
            input_depth = 1
            num_iter = 3001
            figsize = 8
            reg_noise_std = 0.00
            param_noise = True

            if 'skip' in net_type:

                depth = int(net_type[-1])
                if "ce_skip" in net_type:
                    net = ce_skip(in_rotations=1, out_rotations=3,
                              num_input_channels=input_depth,
                              num_output_channels=img_np.shape[0],
                              num_channels_down=[16, 32, 64, 128, 128, 128][:depth],
                              num_channels_up=[16, 32, 64, 128, 128, 128][:depth],
                              num_channels_skip=[0, 0, 0, 0, 0, 0][:depth],
                              filter_size_up=3, filter_size_down=5, filter_skip_size=1,
                              upsample_mode='nearest',  # downsample_mode='avg',
                              need1x1_up=False,
                              need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)
                else:
                    net = skip(input_depth, img_np.shape[0],
                               num_channels_down=[16, 32, 64, 128, 128, 128][:depth],
                               num_channels_up=[16, 32, 64, 128, 128, 128][:depth],
                               num_channels_skip=[0, 0, 0, 0, 0, 0][:depth],
                               filter_size_up=3, filter_size_down=5, filter_skip_size=1,
                               upsample_mode='nearest',  # downsample_mode='avg',
                               need1x1_up=False,
                               need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)

                LR = 0.01

            elif net_type == 'UNET':
                net = UNet(num_input_channels=input_depth, num_output_channels=3,
                           feature_scale=8, more_layers=1,
                           concat_x=False, upsample_mode='deconv',
                           pad=pad, norm_layer=torch.nn.InstanceNorm2d, need_sigmoid=True, need_bias=True)
                LR = 0.001
                param_noise = False

            elif net_type == 'ResNet':
                net = ResNet(input_depth, img_np.shape[0], 8, 32, need_sigmoid=True, act_fun='LeakyReLU')
                LR = 0.001
                param_noise = False

            else:
                assert False
        else:
            assert False
        
        return (net, input_depth, input, img_np, img_mask_np,
                param_noise, reg_noise_std, figsize, out_path,
                OPT_OVER, OPTIMIZER, LR, num_iter)


def main(net_type='skip_depth6', img_name='vase', PLOT = True, show_every=500):

    confs = InpaintingConfigs(net_type, img_name)
    print(confs.net)

    net = confs.net.type(dtype)
    net_input = get_noise(confs.input_depth, confs.input, confs.img_np.shape[1:]).type(dtype)

    # Compute number of parameters
    s = sum(np.prod(list(p.size())) for p in net.parameters())
    print('Number of params: %d' % s)

    # Loss
    mse = torch.nn.MSELoss().type(dtype)

    img_var = np_to_torch(confs.img_np).type(dtype)
    mask_var = np_to_torch(confs.img_mask_np).type(dtype)

    # Main loop

    def closure():
        global i

        if confs.param_noise:
            for n in [x for x in net.parameters() if len(x.size()) == 4]:
                n = n + n.detach().clone().normal_() * n.std() / 50

        net_input = net_input_saved
        if confs.reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * confs.reg_noise_std)

        out = net(net_input)

        total_loss = mse(out * mask_var, img_var * mask_var)
        total_loss.backward()

        print('Iteration %05d    Loss %f' % (i, total_loss.item()), '\r', end='')
        if PLOT and i % show_every == 0:
            out_np = torch_to_np(out)
            plot_image_grid([np.clip(out_np, 0, 1)], factor=confs.figsize, nrow=1)
            out_pil = np_to_pil(out_np)
            print(os.path.join(confs.out_path, "iter_"+str(i)+".png"))
            out_pil.save(os.path.join(confs.out_path, "iter_"+str(i)+".png"))
        i += 1

        return total_loss

    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()

    p = get_params(confs.OPT_OVER, net, net_input)
    optimize(confs.OPTIMIZER, p, closure, confs.LR, confs.num_iter)
    # %%
    out_np = torch_to_np(net(net_input))
    plot_image_grid([out_np], factor=5);
    out_pil = np_to_pil(out_np)
    out_pil.save(os.path.join("final.png"))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        default='skip_depth6',
                        type=str,
                        help='Model name. One of (ce_)skip_depth6|(ce_)skip_depth4|(ce_)skip_depth2|UNET|ResNet')
    parser.add_argument('--image',
                        default='vase',
                        type=str,
                        help='Image name. One of vase|library|kate')
    args = parser.parse_args()
    print(args)
    main(net_type=args.model, img_name=args.image)