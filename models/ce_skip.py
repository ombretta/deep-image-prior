import torch
import torch.nn as nn
from .common import *

def ce_skip(in_rotations=1, out_rotations=3,
        num_input_channels=2, num_output_channels=3, 
        num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128], num_channels_skip=[4, 4, 4, 4, 4], 
        filter_size_down=3, filter_size_up=3, filter_skip_size=1,
        need_sigmoid=True, need_bias=True, 
        pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU',
        need1x1_up=True):
    """Assembles encoder-decoder with skip connections.

    Arguments:
        act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        pad (string): zero|reflection (default: 'zero')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
        downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')

    """
    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

    n_scales = len(num_channels_down) 

    if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)) :
        upsample_mode   = [upsample_mode]*n_scales

    if not (isinstance(downsample_mode, list)or isinstance(downsample_mode, tuple)):
        downsample_mode   = [downsample_mode]*n_scales
    
    if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)) :
        filter_size_down   = [filter_size_down]*n_scales

    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)) :
        filter_size_up   = [filter_size_up]*n_scales

    last_scale = n_scales - 1 

    cur_depth = None

    model = nn.Sequential()
    model_tmp = model

    input_depth = num_input_channels
    for i in range(len(num_channels_down)):

        deeper = nn.Sequential()
        skip = nn.Sequential()

        # skip.add(PrintLayer())
        # deeper.add(PrintLayer())

        if num_channels_skip[i] != 0:
            model_tmp.add(Concat_ce(1, skip, deeper))
        else:
            model_tmp.add(deeper)

        # deeper.add(PrintLayer())
        
        model_tmp.add(bn_3d(num_channels_skip[i] + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i])))

        # deeper.add(PrintLayer())

        if num_channels_skip[i] != 0:
            skip.add(conv_ce(in_rotations, out_rotations, input_depth, num_channels_skip[i],
                             filter_skip_size, bias=need_bias, pad=pad))
            skip.add(bn_3d(num_channels_skip[i]))
            skip.add(act(act_fun))

        # skip.add(Concat_ce(2, GenNoise(nums_noise[i]), skip_part))

        deeper.add(conv_ce(in_rotations, out_rotations, input_depth, num_channels_down[i],
                           filter_size_down[i], stride=2, bias=need_bias, pad=pad))
        in_rotations = out_rotations
        deeper.add(bn_3d(num_channels_down[i]))
        deeper.add(act(act_fun))

        # deeper.add(PrintLayer())

        deeper.add(conv_ce(in_rotations, out_rotations, num_channels_down[i], num_channels_down[i],
                           filter_size_down[i], bias=need_bias, pad=pad))
        deeper.add(bn_3d(num_channels_down[i]))
        deeper.add(act(act_fun))

        # deeper.add(PrintLayer())

        deeper_main = nn.Sequential()

        if i == len(num_channels_down) - 1:
            # The deepest
            k = num_channels_down[i]
        else:
            deeper.add(deeper_main)
            k = num_channels_up[i + 1]

        # Reshape tensor to match the 2d Upsampling function
        deeper.add(View_to_4d(channels=num_channels_down[i], rotations=out_rotations))
        deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))
        deeper.add(View_to_5d(channels=num_channels_down[i], rotations=out_rotations))

        model_tmp.add(conv_ce(in_rotations, out_rotations, num_channels_skip[i] + k, num_channels_up[i],
                              filter_size_up[i], stride=1, bias=need_bias, pad=pad))
        model_tmp.add(bn_3d(num_channels_up[i]))
        model_tmp.add(act(act_fun))

        # model_tmp.add(PrintLayer())

        if need1x1_up:
            model_tmp.add(conv_ce(in_rotations, out_rotations, num_channels_up[i], num_channels_up[i],
                                  1, bias=need_bias, pad=pad))
            model_tmp.add(bn_3d(num_channels_up[i]))
            model_tmp.add(act(act_fun))

        input_depth = num_channels_down[i]
        model_tmp = deeper_main

        # model_tmp.add(PrintLayer())

    model.add(conv_ce(in_rotations, out_rotations, num_channels_up[0], num_output_channels,
                      1, bias=need_bias, pad=pad, groupcosetmaxpool=True))
    if need_sigmoid:
        model.add(nn.Sigmoid())

    return model




def main():
    print("OK")


if __name__=="main":
    main()