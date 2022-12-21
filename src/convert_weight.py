
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Centerface model transform
"""
import os
import torch
from mindspore.train.serialization import load_checkpoint, save_checkpoint
from mindspore import Tensor
import argparse


def load_model(model_path):
    """
    Load model
    """
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)

    return state_dict

def save_model(path, epoch=0, model=None, optimizer=None, state_dict=None):
    """
    Sace model file
    """
    if state_dict is None:
        if isinstance(model, torch.nn.DataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
    data = {'epoch': epoch,
            'state_dict': state_dict}
    if not optimizer is None:
        data['optimizer'] = optimizer.state_dict()
    torch.save(data, path)

def load_model_ms(model_path):
    """
    Load mindspore model
    """
    state_dict_useless = ['global_step', 'learning_rate',
                          'beta1_power', 'beta2_power']
    if os.path.isfile(model_path):
        param_dict = load_checkpoint(model_path)
        param_dict_new = {}
        for key, values in param_dict.items():
            if key in state_dict_useless:
                continue
            else:
                param_dict_new[key] = values
    else:
        assert FileNotFoundError('{} not exists or not a pre-trained file'.format(model_path))
        exit(1)
    return param_dict_new

def name_map(ckpt):
    """
    Name map
    """
    out = {}
    for name in ckpt:
        # conv + bn
        pt_name = name
        # backbone
        pt_name = pt_name.replace('.depthwise_conv', '.depthwise_conv.conv')
        pt_name = pt_name.replace('._depthwise_conv', '._depthwise_conv.conv')
        pt_name = pt_name.replace('.pointwise_conv', '.pointwise_conv.conv')

        pt_name = pt_name.replace('._expand_conv', '._expand_conv.conv')
        pt_name = pt_name.replace('._se_reduce', '._se_reduce.conv')
        pt_name = pt_name.replace('._se_expand', '._se_expand.conv')

        pt_name = pt_name.replace('.p5_down_channel.0', '.p5_down_channel.0.conv')
        pt_name = pt_name.replace('.p4_down_channel.0', '.p4_down_channel.0.conv')
        pt_name = pt_name.replace('.p3_down_channel.0', '.p3_down_channel.0.conv')
        pt_name = pt_name.replace('.p5_to_p6.0', '.p5_to_p6.0.conv')


        pt_name = pt_name.replace('.p4_down_channel_2.0', '.p4_down_channel_2.0.conv')
        pt_name = pt_name.replace('.p5_down_channel_2.0', '.p5_down_channel_2.0.conv')


        pt_name = pt_name.replace('._conv_stem', '._conv_stem.conv')


        pt_name = pt_name.replace('._project_conv', '._project_conv.conv')


        pt_name = pt_name.replace('.moving_mean', '.running_mean')
        pt_name = pt_name.replace('.moving_variance', '.running_var')
        pt_name = pt_name.replace('.gamma', '.weight')
        pt_name = pt_name.replace('.beta', '.bias')


        out[pt_name] = name
    return out

def pt_to_ckpt(pt, ckpt, out_path):
    """
    Pt convert to ckpt file
    """
    state_dict_torch = load_model(pt)
    state_dict_ms = load_model_ms(ckpt)
    name_relate = name_map(state_dict_ms)

    new_params_list = []
    for key in state_dict_torch:
        param_dict = {}
        parameter = state_dict_torch[key]
        parameter = parameter.numpy()

        if "num_batches" in key:
            continue

        param_dict['name'] = name_relate[key]
        param_dict['data'] = Tensor(parameter)
        new_params_list.append(param_dict)

        del state_dict_ms[name_relate[key]]

    save_checkpoint(new_params_list, out_path)
    return state_dict_ms


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="EfficientDet training")
    parser.add_argument("--pt_path", type=str, default="/data/efficientdet_ch/efficientdet-d0.pth")
    parser.add_argument("--ckpt_path", type=str, default="/data/efficientdet_ch/efdet_ms.ckpt")
    parser.add_argument("--out_path", type=str, default="/data/efficientdet_ch/efdet.ckpt")
    args_opt = parser.parse_args()
    pt_to_ckpt(args_opt.pt_path, args_opt.ckpt_path, args_opt.out_path)

