# Fuse Diffusion Features
# Implementation for DIFF for paper 'Diffusion Features to Bridge Domain Gap for Semantic Segmentation'
# Based on HyperFeature
# By Yuxiang Ji

import numpy as np
import torch
from torch import nn
import torch.nn.init as init
from archs.detectron2.resnet import ResNet, BottleneckBlock


class StrideVanillaNetwork(nn.Module):
    def __init__(
            self, 
            feature_dims, 
            device,
            idxs,
            save_timestep=[],
            num_timesteps=None,
            timestep_weight_sharing=False
        ):
        super().__init__()
        self.feature_dims = feature_dims
        self.device = device
        self.save_timestep = save_timestep

        # Count features for each stride, stride id [0, 1, 2]
        self.num_stride = 4
        self.feature_cnts = [0 for _ in range(self.num_stride)]
        self.feature_stride_idx = []
        self.feature_instride_num = []
        for i in range(len(idxs)):
            self.feature_stride_idx.append(idxs[i][0])
            self.feature_instride_num.append(self.feature_cnts[idxs[i][0]])
            self.feature_cnts[idxs[i][0]] += 1

        self.mixing_weights = []
        
        mixing_weights = [torch.ones(self.feature_cnts[i] * len(save_timestep)) * 0.01 for i in range(self.num_stride)]
        self.mixing_weights_stride = nn.ParameterList([nn.Parameter(mixing_weights[i].to(device)) for i in range(self.num_stride)])

        self.in_stride = nn.ParameterList([nn.InstanceNorm2d(feature_dims[i], affine=True) for i in range(len(feature_dims))])

        self.apply(self.weights_init)

    def weights_init(self, m):
        """
        初始化网络权重。
        对于卷积层使用kaiming初始化，对于偏置使用0初始化。
        """
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm) or isinstance(m, nn.InstanceNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)

    def forward(self, batch_list):
        """
        Assumes batch is a list of shape (B, C, H, W) where C is the concatentation of all timesteps features for each layer.

        Features are formed as (1,0)_timestep0,  (1,0)_timestep1, ..., (1,1)_timestep0, (1,1)_timestep1, ... 
        
        Return four features in stride 8, 16, 32, 64
        """

        output_features = [None for _ in range(self.num_stride)]
        mixing_weights_stride = [torch.nn.functional.softmax(self.mixing_weights_stride[i], dim=0) for i in range(self.num_stride)]
        # print(mixing_weights_stride[0], mixing_weights_stride[1], mixing_weights_stride[2])

        for idx_i in range(len(self.feature_stride_idx)):
            for timestep_i in range(len(self.save_timestep)):
                stride_id = self.feature_stride_idx[idx_i]
                instride_num = self.feature_instride_num[idx_i]

                # Chunk the batch according the layer
                # Account for looping if there are multiple timesteps
                num_channel = self.feature_dims[idx_i]
                start_channel = instride_num * len(self.save_timestep) * num_channel + timestep_i * num_channel
                feats = batch_list[stride_id][:, start_channel:start_channel+num_channel, :, :]

                feats = self.in_stride[idx_i](feats)

                feats = mixing_weights_stride[stride_id][timestep_i * self.feature_cnts[stride_id] + instride_num] * feats
                if output_features[stride_id] is None:
                    output_features[stride_id] = feats
                else:
                    output_features[stride_id] += feats

        return output_features[3], output_features[2], output_features[1], output_features[0]


class StrideDirectAggregationNetwork2(nn.Module):
    def __init__(
            self, 
            feature_dims_by_stride,
            feature_dims_by_idx, 
            device,
            idxs,
            projection_dim=[768, 384, 192, 96],  ## Stride 64, 32, 16, 8
            num_norm_groups=32,
            num_res_blocks=1,
            save_timestep=[],
            num_timesteps=None,
            timestep_weight_sharing=False
        ):
        super().__init__()
        self.bottleneck_layers = nn.ModuleList()
        self.aggregation_stride_layers = nn.ModuleList()
        self.feature_dims_by_stride = feature_dims_by_stride
        self.feature_dims_by_idx = feature_dims_by_idx
        # For CLIP symmetric cross entropy loss during training
        self.logit_scale = torch.ones([]) * np.log(1 / 0.07)
        self.device = device
        self.save_timestep = save_timestep

        # Count features for each stride, stride id [0, 1, 2]
        self.num_stride = len(projection_dim)
        self.feature_cnts = [0 for _ in range(self.num_stride)]
        self.feature_stride_idx = []
        self.feature_instride_num = []
        for i in range(len(idxs)):
            self.feature_stride_idx.append(idxs[i][0])
            self.feature_instride_num.append(self.feature_cnts[idxs[i][0]])
            self.feature_cnts[idxs[i][0]] += 1

        for l, feature_dim in enumerate(self.feature_dims_by_idx):
            stride_id = self.feature_stride_idx[l]

            bottleneck_layer = nn.Sequential(
                *ResNet.make_stage(
                    BottleneckBlock,
                    num_blocks=num_res_blocks,
                    in_channels=feature_dim,
                    bottleneck_channels=projection_dim[stride_id] // 4,
                    out_channels=projection_dim[stride_id],
                    norm="GN",
                    num_norm_groups=num_norm_groups
                )
            )
            self.bottleneck_layers.append(bottleneck_layer)

        for stride_id in range(self.num_stride):
            bottleneck_layer = nn.Sequential(
                *ResNet.make_stage(
                    BottleneckBlock,
                    num_blocks=num_res_blocks,
                    in_channels=projection_dim[stride_id] * 3 * len(save_timestep),
                    bottleneck_channels=projection_dim[stride_id] // 4,
                    out_channels=projection_dim[stride_id],
                    norm="GN",
                    num_norm_groups=num_norm_groups
                )
            )
            self.aggregation_stride_layers.append(bottleneck_layer)
        
        self.bottleneck_layers = self.bottleneck_layers.to(device)
        self.aggregation_stride_layers = self.aggregation_stride_layers.to(device)

        self.apply(self.weights_init)

    def weights_init(self, m):
        """
        初始化网络权重。
        对于卷积层使用kaiming初始化，对于偏置使用0初始化。
        """
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)

    def forward(self, batch_list):
        """
        Assumes batch is a list of shape (B, C, H, W) where C is the concatentation of all timesteps features for each layer.

        Features are formed as (1,0)_timestep0,  (1,0)_timestep1, ..., (1,1)_timestep0, (1,1)_timestep1, ... 
        
        Return four features in stride 8, 16, 32, 64
        """

        intermediate_features = [None for _ in range(self.num_stride)]
        output_features = [None for _ in range(self.num_stride)]

        for idx_i in range(len(self.feature_stride_idx)):
            start_channel = 0
            for timestep_i in range(len(self.save_timestep)):
                stride_id = self.feature_stride_idx[idx_i]
                if batch_list[stride_id] == None:
                    continue
                
                # print('scgscg', stride_id, timestep_i, self.feature_dims_by_idx[idx_i], batch_list[stride_id].shape)
                instride_num = self.feature_instride_num[idx_i]

                # Share bottleneck layers across timesteps
                bottleneck_layer = self.bottleneck_layers[idx_i]
                # Chunk the batch according the layer
                # Account for looping if there are multiple timesteps
                num_channel = self.feature_dims_by_idx[idx_i]
                # start_channel = instride_num * len(self.save_timestep) * num_channel + timestep_i * num_channel
                feats = batch_list[stride_id][:, start_channel:start_channel+num_channel, :, :]
                # print(batch_list[stride_id].shape)
                # print(idx_i, f'timestep_i={timestep_i}, stride_id={stride_id}, instride_num={instride_num}, l={start_channel}, r={start_channel+num_channel}')
                # print(timestep_i * self.feature_cnts[stride_id] + instride_num)
                # Downsample the number of channels and weight the layer
                # print(stride_id, instride_num, torch.any(torch.isnan(feats)))
                bottlenecked_feature = bottleneck_layer(feats)
                
                if intermediate_features[stride_id] is None:
                    intermediate_features[stride_id] = bottlenecked_feature
                else:
                    intermediate_features[stride_id] = torch.cat((intermediate_features[stride_id], bottlenecked_feature), dim=1)
                start_channel = start_channel + num_channel
        
        for stride_id in range(4):
            output_features[stride_id] = self.aggregation_stride_layers[stride_id](intermediate_features[stride_id])

        return output_features[3], output_features[2], output_features[1], output_features[0]



class StrideDirectAggregationNetwork(nn.Module):
    def __init__(
            self, 
            feature_dims_by_idx,
            feature_dims_by_stride, 
            device,
            idxs,
            projection_dim=[768, 384, 192, 96],  ## Stride 64, 32, 16, 8
            num_norm_groups=16,
            num_res_blocks=1,
            save_timestep=[],
            num_timesteps=None,
            timestep_weight_sharing=False
        ):
        super().__init__()
        self.bottleneck_layers = nn.ModuleList()
        self.feature_dims_by_stride = feature_dims_by_stride
        # For CLIP symmetric cross entropy loss during training
        self.logit_scale = torch.ones([]) * np.log(1 / 0.07)
        self.device = device
        self.save_timestep = save_timestep

        # Count features for each stride, stride id [0, 1, 2]
        self.num_stride = len(projection_dim)
        self.feature_cnts = [0 for _ in range(self.num_stride)]
        self.feature_stride_idx = []
        self.feature_instride_num = []
        for i in range(len(idxs)):
            self.feature_stride_idx.append(idxs[i][0])
            self.feature_instride_num.append(self.feature_cnts[idxs[i][0]])
            self.feature_cnts[idxs[i][0]] += 1

        for stride_id in range(self.num_stride):
            # print(stride_id, feature_dims[stride_id * 3] * 3 * len(save_timestep), projection_dim[stride_id])
            bottleneck_layer = nn.Sequential(
                *ResNet.make_stage(
                    BottleneckBlock,
                    num_blocks=num_res_blocks,
                    in_channels=feature_dims_by_stride[stride_id] * len(save_timestep),
                    bottleneck_channels=projection_dim[stride_id] // 4,
                    out_channels=projection_dim[stride_id],
                    norm="GN",
                    num_norm_groups=num_norm_groups
                )
            )
            self.bottleneck_layers.append(bottleneck_layer)
        
        self.bottleneck_layers = self.bottleneck_layers.to(device)

        self.apply(self.weights_init)

    def weights_init(self, m):
        """
        初始化网络权重。
        对于卷积层使用kaiming初始化，对于偏置使用0初始化。
        """
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)

    def forward(self, batch_list):
        """
        Assumes batch is a list of shape (B, C, H, W) where C is the concatentation of all timesteps features for each layer.

        Features are formed as (1,0)_timestep0,  (1,0)_timestep1, ..., (1,1)_timestep0, (1,1)_timestep1, ... 
        
        Return four features in stride 8, 16, 32, 64
        """

        output_features = [None for _ in range(self.num_stride)]
        for stride_id in range(self.num_stride):
            # print(stride_id, batch_list[stride_id].shape)
            feats = batch_list[stride_id]
            if feats == None:
                continue
            output_features[stride_id] = self.bottleneck_layers[stride_id](feats)

        return output_features[3], output_features[2], output_features[1], output_features[0]


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2  # Keep the spatial size unchanged

        self.conv = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,  # 4 gates: input, forget, cell, output
            kernel_size=kernel_size,
            padding=self.padding
        )

        self.init_weights()

    def forward(self, x, hidden_state):
        h_t, c_t = hidden_state

        combined = torch.cat((x, h_t), dim=1)
        gates = self.conv(combined)

        i_t, f_t, g_t, o_t = torch.chunk(gates, 4, dim=1)

        i_t = torch.sigmoid(i_t)
        f_t = torch.sigmoid(f_t)
        g_t = torch.tanh(g_t)
        o_t = torch.sigmoid(o_t)

        c_t = f_t * c_t + i_t * g_t
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t
    
    def init_weights(self):
        for name, param in self.conv.named_parameters():
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                if 'forget' in name:  # Forget gate weights initialization
                    init.constant_(param, 1.0)
                else:
                    init.xavier_normal_(param)

class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super(ConvLSTM, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.cell = ConvLSTMCell(input_channels, hidden_channels, kernel_size)

    def forward(self, x):
        batch_size, _, _, height, width = x.size()
        h_t = torch.zeros((batch_size, self.hidden_channels, height, width), device=x.device)
        c_t = torch.zeros((batch_size, self.hidden_channels, height, width), device=x.device)
        
        output = []
        for t in range(x.size(1)):
            h_t, c_t = self.cell(x[:, t, :, :, :], (h_t, c_t))
            output.append(h_t)

        return h_t

class StrideRNNNetwork(nn.Module):
    def __init__(
            self, 
            feature_dims_by_idx,
            feature_dims_by_stride, 
            device,
            idxs,
            projection_dim=[768, 384, 192, 96],  ## Stride 64, 32, 16, 8
            num_norm_groups=32,
            num_res_blocks=1,
            save_timestep=[],
            num_timesteps=None,
            timestep_weight_sharing=False
        ):
        super().__init__()
        self.bottleneck_layers = nn.ModuleList()
        self.rnn_layers = nn.ModuleList()
        self.out_layers = nn.ModuleList()
        self.feature_dims = feature_dims_by_idx
        self.device = device
        self.save_timestep = save_timestep

        # Count features for each stride, stride id [0, 1, 2]
        self.num_stride = len(projection_dim)
        self.feature_cnts = [0 for _ in range(self.num_stride)]
        self.feature_stride_idx = []
        self.feature_instride_num = []
        for i in range(len(idxs)):
            self.feature_stride_idx.append(idxs[i][0])
            self.feature_instride_num.append(self.feature_cnts[idxs[i][0]])
            self.feature_cnts[idxs[i][0]] += 1

        for l, feature_dim in enumerate(self.feature_dims):
            stride_id = self.feature_stride_idx[l]

            bottleneck_layer = nn.Sequential(
                *ResNet.make_stage(
                    BottleneckBlock,
                    num_blocks=num_res_blocks,
                    in_channels=feature_dim,
                    bottleneck_channels=projection_dim[stride_id] // 4,
                    out_channels=projection_dim[stride_id],
                    norm="GN",
                    num_norm_groups=num_norm_groups
                )
            )
            self.bottleneck_layers.append(bottleneck_layer)

            rnn_layer = ConvLSTM(input_channels=projection_dim[stride_id], hidden_channels=projection_dim[stride_id])
            self.rnn_layers.append(rnn_layer)
        
        for stride_id in range(4):
            out_layer = nn.Sequential(
                *ResNet.make_stage(
                    BottleneckBlock,
                    num_blocks=num_res_blocks,
                    in_channels=projection_dim[stride_id] * 3,
                    bottleneck_channels=projection_dim[stride_id] // 4,
                    out_channels=projection_dim[stride_id],
                    norm="GN",
                    num_norm_groups=num_norm_groups
                )
            )
            self.out_layers.append(out_layer)

    def forward(self, batch_list):
        """
        Assumes batch is a list of shape (B, C, H, W) where C is the concatentation of all timesteps features for each layer.

        Features are formed as (1,0)_timestep0,  (1,0)_timestep1, ..., (1,1)_timestep0, (1,1)_timestep1, ... 
        
        Return four features in stride 8, 16, 32, 64
        """

        output_features = [[] for _ in range(self.num_stride)]

        for idx_i in range(len(self.feature_stride_idx)):
            feats_idx = None
            bottleneck_layer = self.bottleneck_layers[idx_i]
            rnn_layer = self.rnn_layers[idx_i]
            stride_id = self.feature_stride_idx[idx_i]
            instride_num = self.feature_instride_num[idx_i]
            for timestep_i in range(len(self.save_timestep)):
                
                # Share bottleneck layers across timesteps
                # Chunk the batch according the layer
                # Account for looping if there are multiple timesteps
                num_channel = self.feature_dims[idx_i]
                start_channel = instride_num * len(self.save_timestep) * num_channel + timestep_i * num_channel
                feats = batch_list[stride_id][:, start_channel:start_channel+num_channel, :, :]
                # print(batch_list[stride_id].shape)
                # print(idx_i, f'timestep_i={timestep_i}, stride_id={stride_id}, instride_num={instride_num}, l={start_channel}, r={start_channel+num_channel}')
                # print(timestep_i * self.feature_cnts[stride_id] + instride_num)
                # Downsample the number of channels and weight the layer
                # print(stride_id, instride_num, torch.any(torch.isnan(feats)))
                bottlenecked_feature = bottleneck_layer(feats)
                if feats_idx == None:
                    feats_idx = bottlenecked_feature[: , None, ...]
                else:
                    feats_idx = torch.cat((feats_idx, bottlenecked_feature[: , None, ...]), dim=1)
            output_features[stride_id].append(rnn_layer(feats_idx))
        for stride_id in range(4):
            output_features[stride_id] = self.out_layers[stride_id](torch.cat(output_features[stride_id], dim=1))
        return output_features[3], output_features[2], output_features[1], output_features[0]


class StrideAggregationNetwork(nn.Module):
    """
    Module for aggreagating feature maps across time for diffrent strides (8, 16, 32).
    """
    def __init__(
            self, 
            feature_dims_by_idx,
            feature_dims_by_stride, 
            device,
            idxs,
            projection_dim=[768, 384, 192, 96],  ## Stride 64, 32, 16, 8
            num_norm_groups=16,
            num_res_blocks=1,
            save_timestep=[],
            num_timesteps=None,
            timestep_weight_sharing=False
        ):
        super().__init__()
        self.bottleneck_layers = nn.ModuleList()
        self.feature_dims_by_idx = feature_dims_by_idx
        # print('scgscg', self.feature_dims_by_idx)
        # For CLIP symmetric cross entropy loss during training
        self.logit_scale = torch.ones([]) * np.log(1 / 0.07)
        self.device = device
        self.save_timestep = save_timestep

        # Count features for each stride, stride id [0, 1, 2]
        self.num_stride = len(projection_dim)
        self.feature_cnts = [0 for _ in range(self.num_stride)]
        self.feature_stride_idx = []
        self.feature_instride_num = []

        for i in range(len(idxs)):
            self.feature_stride_idx.append(idxs[i][0])
            self.feature_instride_num.append(self.feature_cnts[idxs[i][0]])
            self.feature_cnts[idxs[i][0]] += 1

        self.mixing_weights = []
        # self.mixing_weights_names = []
        for l, feature_dim in enumerate(self.feature_dims_by_idx):
            stride_id = self.feature_stride_idx[l]

            bottleneck_layer = nn.Sequential(
                *ResNet.make_stage(
                    BottleneckBlock,
                    num_blocks=num_res_blocks,
                    in_channels=feature_dim,
                    bottleneck_channels=projection_dim[stride_id] // 4,
                    out_channels=projection_dim[stride_id],
                    norm="GN",
                    num_norm_groups=num_norm_groups
                )
            )
            self.bottleneck_layers.append(bottleneck_layer)
        
        self.bottleneck_layers = self.bottleneck_layers.to(device)
        mixing_weights = [torch.ones(self.feature_cnts[i] * len(save_timestep)) for i in range(self.num_stride)]
        self.mixing_weights_stride = nn.ParameterList([nn.Parameter(mixing_weights[i].to(device)) for i in range(self.num_stride)])
        # self.mixing_weights_stride.requires_grad = False

        self.apply(self.weights_init)

    def weights_init(self, m):
        """
        初始化网络权重。
        对于卷积层使用kaiming初始化，对于偏置使用0初始化。
        """
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)

    def forward(self, batch_list):
        """
        Assumes batch is a list of shape (B, C, H, W) where C is the concatentation of all timesteps features for each layer.

        Features are formed as (1,0)_timestep0,  (1,0)_timestep1, ..., (1,1)_timestep0, (1,1)_timestep1, ... 
        
        Return four features in stride 8, 16, 32, 64
        """
        # print('jyxjyx aggregation', batch_list[0].shape, batch_list[1].shape, batch_list[2].shape, batch_list[3].shape)

        output_features = [None for _ in range(self.num_stride)]
        mixing_weights_stride = [torch.nn.functional.softmax(self.mixing_weights_stride[i], dim=0) for i in range(self.num_stride)]
        # print(mixing_weights_stride[0], mixing_weights_stride[1], mixing_weights_stride[2])


        for idx_i in range(len(self.feature_stride_idx)):
            start_channel = 0
            for timestep_i in range(len(self.save_timestep)):
                stride_id = self.feature_stride_idx[idx_i]
                if batch_list[stride_id] == None:
                    continue
                
                # print('scgscg', stride_id, timestep_i, self.feature_dims_by_idx[idx_i], batch_list[stride_id].shape)
                instride_num = self.feature_instride_num[idx_i]

                # Share bottleneck layers across timesteps
                bottleneck_layer = self.bottleneck_layers[idx_i]
                # Chunk the batch according the layer
                # Account for looping if there are multiple timesteps
                num_channel = self.feature_dims_by_idx[idx_i]
                # start_channel = instride_num * len(self.save_timestep) * num_channel + timestep_i * num_channel
                feats = batch_list[stride_id][:, start_channel:start_channel+num_channel, :, :]
                # print(batch_list[stride_id].shape)
                # print(idx_i, f'timestep_i={timestep_i}, stride_id={stride_id}, instride_num={instride_num}, l={start_channel}, r={start_channel+num_channel}')
                # print(timestep_i * self.feature_cnts[stride_id] + instride_num)
                # Downsample the number of channels and weight the layer
                # print(stride_id, instride_num, torch.any(torch.isnan(feats)))
                bottlenecked_feature = bottleneck_layer(feats)
                bottlenecked_feature = mixing_weights_stride[stride_id][timestep_i * self.feature_cnts[stride_id] + instride_num] * bottlenecked_feature
                if output_features[stride_id] is None:
                    output_features[stride_id] = bottlenecked_feature
                else:
                    output_features[stride_id] += bottlenecked_feature
                start_channel = start_channel + num_channel

        return output_features[3], output_features[2], output_features[1], output_features[0]


class AggregationNetwork(nn.Module):
    """
    Module for aggregating feature maps across time and space.
    Design inspired by the Feature Extractor from ODISE (Xu et. al., CVPR 2023).
    https://github.com/NVlabs/ODISE/blob/5836c0adfcd8d7fd1f8016ff5604d4a31dd3b145/odise/modeling/backbone/feature_extractor.py
    """
    def __init__(
            self, 
            feature_dims, 
            device, 
            projection_dim=384, 
            num_norm_groups=32,
            num_res_blocks=1, 
            save_timestep=[],
            num_timesteps=None,
            timestep_weight_sharing=False
        ):
        super().__init__()
        self.bottleneck_layers = nn.ModuleList()
        self.feature_dims = feature_dims    
        # For CLIP symmetric cross entropy loss during training
        self.logit_scale = torch.ones([]) * np.log(1 / 0.07)
        self.device = device
        self.save_timestep = save_timestep

        self.mixing_weights_names = []
        for l, feature_dim in enumerate(self.feature_dims):
            bottleneck_layer = nn.Sequential(
                *ResNet.make_stage(
                    BottleneckBlock,
                    num_blocks=num_res_blocks,
                    in_channels=feature_dim,
                    bottleneck_channels=projection_dim // 4,
                    out_channels=projection_dim,
                    norm="GN",
                    num_norm_groups=num_norm_groups
                )
            )
            self.bottleneck_layers.append(bottleneck_layer)
            for t in save_timestep:
                # 1-index the layer name following prior work
                self.mixing_weights_names.append(f"timestep-{save_timestep}_layer-{l+1}")
        
        self.bottleneck_layers = self.bottleneck_layers.to(device)
        mixing_weights = torch.ones(len(self.bottleneck_layers) * len(save_timestep))
        self.mixing_weights = nn.Parameter(mixing_weights.to(device))

    def forward(self, batch):
        """
        Assumes batch is shape (B, C, H, W) where C is the concatentation of all layer features.
        """
        output_feature = None
        start = 0
        mixing_weights = torch.nn.functional.softmax(self.mixing_weights)
        for i in range(len(mixing_weights)):
            # Share bottleneck layers across timesteps
            bottleneck_layer = self.bottleneck_layers[i % len(self.feature_dims)]
            # Chunk the batch according the layer
            # Account for looping if there are multiple timesteps
            end = start + self.feature_dims[i % len(self.feature_dims)]
            feats = batch[:, start:end, :, :]
            start = end
            # Downsample the number of channels and weight the layer
            bottlenecked_feature = bottleneck_layer(feats)
            bottlenecked_feature = mixing_weights[i] * bottlenecked_feature
            if output_feature is None:
                output_feature = bottlenecked_feature
            else:
                output_feature += bottlenecked_feature
        return output_feature