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


def main(net_type='skip_depth6', img_name='vase'):

    PLOT = True
    imsize = -1
    dim_div_by = 64

    # Choose net type
    NET_TYPE = net_type  # one of skip_depth6|skip_depth4|skip_depth2|UNET|ResNet

    # Choose figure
    IMG_NAME = img_name #'vase' #'library' #'kate'
    img_path = 'data/inpainting/'+IMG_NAME+'.png'
    mask_path = 'data/inpainting/'+IMG_NAME+'_mask.png'

    # Define output path
    out_path = 'results/inpainting/'+NET_TYPE+'/'+IMG_NAME+'/'
    if not os.path.exists('results/inpainting/'+NET_TYPE):
        os.mkdir('results/inpainting/'+NET_TYPE)
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    # Load image and mask
    img_np, img_mask_np = load_image_and_mask(img_path, mask_path, imsize, dim_div_by)

    # Setup
    pad = 'reflection'  # 'zero'
    OPT_OVER = 'net'
    OPTIMIZER = 'adam'

    if 'vase.png' in img_path:
        INPUT = 'meshgrid'
        input_depth = 2
        LR = 0.01
        num_iter = 5001
        param_noise = False
        show_every = 50
        figsize = 5
        reg_noise_std = 0.03

        net = skip(input_depth, img_np.shape[0],
                   num_channels_down=[128] * 5,
                   num_channels_up=[128] * 5,
                   num_channels_skip=[0] * 5,
                   upsample_mode='nearest', filter_skip_size=1, filter_size_up=3, filter_size_down=3,
                   need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)

        print(net)

    elif ('kate.png' in img_path) or ('peppers.png' in img_path):
        # Same params and net as in super-resolution and denoising
        INPUT = 'noise'
        input_depth = 32
        LR = 0.01
        num_iter = 6001
        param_noise = False
        show_every = 50
        figsize = 5
        reg_noise_std = 0.03

        net = skip(input_depth, img_np.shape[0],
                   num_channels_down=[128] * 5,
                   num_channels_up=[128] * 5,
                   num_channels_skip=[128] * 5,
                   filter_size_up=3, filter_size_down=3,
                   upsample_mode='nearest', filter_skip_size=1,
                   need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)

    elif 'library.png' in img_path:

        INPUT = 'noise'
        input_depth = 1

        num_iter = 3001
        show_every = 50
        figsize = 8
        reg_noise_std = 0.00
        param_noise = True

        if 'skip' in NET_TYPE:

            depth = int(NET_TYPE[-1])
            net = skip(input_depth, img_np.shape[0],
                       num_channels_down=[16, 32, 64, 128, 128, 128][:depth],
                       num_channels_up=[16, 32, 64, 128, 128, 128][:depth],
                       num_channels_skip=[0, 0, 0, 0, 0, 0][:depth],
                       filter_size_up=3, filter_size_down=5, filter_skip_size=1,
                       upsample_mode='nearest',  # downsample_mode='avg',
                       need1x1_up=False,
                       need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)

            LR = 0.01

        elif NET_TYPE == 'UNET':

            net = UNet(num_input_channels=input_depth, num_output_channels=3,
                       feature_scale=8, more_layers=1,
                       concat_x=False, upsample_mode='deconv',
                       pad='zero', norm_layer=torch.nn.InstanceNorm2d, need_sigmoid=True, need_bias=True)

            LR = 0.001
            param_noise = False

        elif NET_TYPE == 'ResNet':

            net = ResNet(input_depth, img_np.shape[0], 8, 32, need_sigmoid=True, act_fun='LeakyReLU')

            LR = 0.001
            param_noise = False

        else:
            assert False
    else:
        assert False

    net = net.type(dtype)
    net_input = get_noise(input_depth, INPUT, img_np.shape[1:]).type(dtype)

    print("Net input 0", net_input.shape)

    # Compute number of parameters
    s = sum(np.prod(list(p.size())) for p in net.parameters())
    print('Number of params: %d' % s)

    # Loss
    mse = torch.nn.MSELoss().type(dtype)

    img_var = np_to_torch(img_np).type(dtype)
    mask_var = np_to_torch(img_mask_np).type(dtype)

    # Main loop

    def closure():
        global i

        if param_noise:
            for n in [x for x in net.parameters() if len(x.size()) == 4]:
                n = n + n.detach().clone().normal_() * n.std() / 50

        print("Net input saved", net_input_saved.shape)

        net_input = net_input_saved
        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)

        print("Net input", net_input.shape)
        out = net(net_input)

        print("Net output", out.shape)

        total_loss = mse(out * mask_var, img_var * mask_var)
        total_loss.backward()

        print('Iteration %05d    Loss %f' % (i, total_loss.item()), '\r', end='')
        if PLOT and i % show_every == 0:
            out_np = torch_to_np(out)
            plot_image_grid([np.clip(out_np, 0, 1)], factor=figsize, nrow=1)
            out_pil = np_to_pil(out_np)
            out_pil.save(os.path.join(out_path, "iter_"+str(i)+".png"))
        i += 1

        return total_loss

    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()

    p = get_params(OPT_OVER, net, net_input)
    optimize(OPTIMIZER, p, closure, LR, num_iter)
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