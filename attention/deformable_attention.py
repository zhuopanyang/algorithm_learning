# -*- coding: utf-8 -*
import torch
import torch.nn.functional as F
import torch.nn as nn


def multi_scale_deformable_attn_pytorch(value, spatial_shapes, sampling_locations, attention_weights):
    batch, _, num_head, embeding_dim_perhead = value.shape
    _, query_size, _, level_num, sample_num, _ = sampling_locations.shape
    split_list = []
    for h, w in spatial_shapes:
        split_list.append(int(h * w))
    value_list = value.split(split_size=tuple(split_list), dim=1)

    sampling_grid = 2 * sampling_locations - 1
    output_list = []
    for level_id, (h, w) in enumerate(spatial_shapes):
        value_l = value_list[level_id].permute(0, 2, 3, 1).view(batch * num_head, embeding_dim_perhead, h, w)
        sampling_grid_l = sampling_grid[:, :, :, level_id, :, :].permute(0, 2, 1, 3, 4).view(batch * num_head,
                                                                                             query_size, sample_num, 2)
        output = F.grid_sample(input=value_l, grid=sampling_grid_l, mode='bilinear', padding_mode='zeros',
                               align_corners=False)
        output_list.append(output)

    outputs = torch.stack(output_list, dim=-2)
    attention_weights = attention_weights.permute(0, 2, 1, 3, 4).view(batch * num_head, 1, query_size, level_num,
                                                                      sample_num)
    outputs = outputs * attention_weights

    outputs = outputs.sum(-1).sum(-1).view(batch, num_head, embeding_dim_perhead, query_size).permute(0, 3, 1, 2).view(
        batch, query_size, num_head * embeding_dim_perhead)
    return outputs.contiguous()
