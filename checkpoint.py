#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
from shutil import copy2, move

import torch

def init_model_from_weights(
    model,
    state_dict,
    skip_layers=None,
    print_init_layers=True,
):
    state_dict = state_dict["model"]

    all_layers = model.state_dict()
    init_layers = {layername: False for layername in all_layers}

    new_state_dict = {}
    for param_name in state_dict:
        if "module.trunk.0" not in param_name:
            continue
        param_data = param_name.split(".")
        newname = "backbone_net"
        for i in range(len(param_data[3:])):
            newname += "." + param_data[i+3]
        new_state_dict[newname] = state_dict[param_name]

    state_dict = new_state_dict
            
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    not_found, not_init = [], []
    for layername in all_layers.keys():
        if (
            skip_layers and len(skip_layers) > 0 and layername.find(skip_layers) >= 0
        ) or layername.find("num_batches_tracked") >= 0:
            if print_init_layers and (local_rank == 0):
                not_init.append(layername)
                print(f"Ignored layer:\t{layername}")
            continue
        if layername in state_dict:
            param = state_dict[layername]
            if not isinstance(param, torch.Tensor):
                param = torch.from_numpy(param)
            # if param.shape != all_layers[layername].shape:
            #     print(f"Ignore layer: \t{layername}")
            else:
                if (param.shape != all_layers[layername].shape):
                    param = handle_shape(param, all_layers[layername])
                all_layers[layername].copy_(param)
                init_layers[layername] = True
                if print_init_layers and (local_rank == 0):
                    print(f"Init layer:\t{layername}")
        else:
            not_found.append(layername)
            if print_init_layers and (local_rank == 0):
                print(f"Not found:\t{layername}")
    ####################### DEBUG ############################
    # _print_state_dict_shapes(model.state_dict())
    torch.cuda.empty_cache()
    return model

def handle_shape(source, dst_shape):
    print(f"Convert {source.shape} to {dst_shape.shape}")
    if (source.shape == torch.Size([64,3,1,1])):
        temp_tensor = torch.zeros(dst_shape.shape)  # Initialize with zeros
        temp_tensor[:, :3, :, :] = source
        return temp_tensor
    if (source.shape == torch.Size([512,512,1,1]) and dst_shape.shape == torch.Size([256,512,1,1])):
        temp_tensor = source[:256,:,:,:]
        return temp_tensor
    if (source.shape == torch.Size([512]) and dst_shape.shape == torch.Size([256])):
        temp_tensor = source[:256]
        return temp_tensor
    if (source.shape == torch.Size([512,512,1,1]) and dst_shape.shape == torch.Size([256,256,1,1])):
        temp_tensor = source[:256,:256,:,:]
        return temp_tensor
    if (source.shape == torch.Size([512,768,1,1]) and dst_shape.shape == torch.Size([256,512,1,1])):
        temp_tensor = source[:256,:512,:,:]
        return temp_tensor