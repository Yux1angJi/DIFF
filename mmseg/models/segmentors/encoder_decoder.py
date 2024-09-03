# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications:
# - Support for seg_weight and forward_with_aux
# - Add forward_style

# Implementation for IPKL for paper 'Diffusion Features to Bridge Domain Gap for Semantic Segmentation'
# By Yuxiang Ji

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor

import numpy as np
from PIL import Image

from mmseg.models.decode_heads.daformer_head import DAFormerHead


@SEGMENTORS.register_module()
class EncoderDecoder(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 diff_train=True):
        super(EncoderDecoder, self).__init__(init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = builder.build_backbone(backbone)

        self.diff_train = diff_train
        self.iter = 0
        self.total_iter = 40000

        if self.backbone.__class__.__name__ == 'DIFF':
            self.backbone_name = 'DIFF'
        else:
            self.backbone_name = 'others'
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)

    def extract_feat(self, img, gt_semantic_seg=None, file_name=None):
        """Extract features from images."""
        if self.backbone_name == 'DIFF':
            # import cv2
            # # file_name = file_name.replace('_leftImg8bit.png', '_gtFine_labelTrainIds.png').replace('img', 'gt')
            # file_name = file_name.replace('.png', '_labelTrainIds.png').replace('img', 'gt')
            # input_label = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
            # input_label = cv2.resize(input_label, (512, 512), interpolation=cv2.INTER_NEAREST)
            # gt_semantic_seg = torch.Tensor(input_label)[None, ...].cuda()
            # print('jyxjyxjyx gt', gt_semantic_seg.shape)
            x = self.backbone(img, gt_semantic_seg)
        else:
            x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def generate_pseudo_label(self, img, img_metas):
        return self.encode_decode(img, img_metas)

    def encode_decode(self, img, img_metas, upscale_pred=True, gt_semantic_seg=None):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img, gt_semantic_seg, file_name=img_metas[0]['filename'])
        # x = self.extract_feat(img, gt_semantic_seg)
        out = self._decode_head_forward_test(x, img_metas)
        if upscale_pred:
            out = resize(
                input=out,
                size=img.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        return out

    def forward_with_aux(self, img, img_metas):
        ret = {}

        x = self.extract_feat(img)
        out = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        ret['main'] = out

        if self.with_auxiliary_head:
            assert not isinstance(self.auxiliary_head, nn.ModuleList)
            out_aux = self.auxiliary_head.forward_test(x, img_metas,
                                                       self.test_cfg)
            out_aux = resize(
                input=out_aux,
                size=img.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            ret['aux'] = out_aux

        return ret

    def _decode_head_forward_train(self,
                                   x,
                                   img_metas,
                                   gt_semantic_seg,
                                   seg_weight=None,
                                   return_logits=False,
                                   prefix='decode'):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, img_metas,
                                                    gt_semantic_seg,
                                                    self.train_cfg,
                                                    seg_weight, return_logits)

        losses.update(add_prefix(loss_decode, prefix))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def _auxiliary_head_forward_train(self,
                                      x,
                                      img_metas,
                                      gt_semantic_seg,
                                      seg_weight=None):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, img_metas,
                                                  gt_semantic_seg,
                                                  self.train_cfg, seg_weight)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def mask_random_pixels(self, imgs, gt_semantic_segs, mask_prob=0.25):
        """
        随机将图像中的像素置为0。
        :param images: 输入的图像批次，形状为 (B, C, H, W)
        :param mask_prob: 要被置为0的像素比例
        :return: 处理后的图像批次
        """
        # 生成一个随机mask，其中一些元素会被设置为False
        random_mask = torch.rand((imgs.shape[2], imgs.shape[3])).to(imgs.device) > mask_prob  # True表示保留像素，False表示置0

        imgs = imgs * random_mask.float()

        gt_semantic_segs = gt_semantic_segs * random_mask + (~random_mask) * 255

        return imgs, gt_semantic_segs  # 用mask乘以原图，False位置变为0

    def l1_loss(self, input1, input2, lamb=0.1):
        l1_loss = F.l1_loss(input1, input2)
        return {'consistency_loss': l1_loss.mean()*lamb}

    def kl_loss(self, pred_logits, ref_logits, lamb=0.1, temp_s=5.0, temp_t=5.0, max_v=1.0):
        temp = temp_s - (self.iter / self.total_iter) * (temp_s - temp_t)
        log_probs_pred = F.log_softmax(pred_logits / temp, dim=1)
        probs_ref = F.softmax(ref_logits / temp, dim=1)
        kl_loss = F.kl_div(log_probs_pred, probs_ref, reduction='batchmean', log_target=False).mean() * lamb * temp * temp
        kl_loss = torch.clamp(kl_loss, min=-float('inf'), max=max_v)
        return {'consistency_loss': kl_loss}
    
    def ce_loss(self, pred_logits, ref_logits, lamb=0.1):
        ref_target = ref_logits.argmax(dim=1)
        ce_loss = F.cross_entropy(pred_logits, ref_target, ignore_index=255)
        return {'consistency_loss': ce_loss.mean()*lamb}
    
    def mse_loss(self, input1, input2, lamb=0.1):
        mse_loss = F.mse_loss(input1, input2)
        return {'consistency_loss': mse_loss.mean()*lamb}
    
    def mse_loss_stride(self, input1, input2, lamb=0.1):
        mse_loss = 0.0
        for i in range(4):
            mse_loss += F.mse_loss(input1[i], input2[i]) * lamb
        return {'feature_consistency_loss': mse_loss.mean()}
    
    def dice_loss(self, input1, input2, smooth=1.0, lamb=0.1):
        # 将 logits 转换为概率分布
        input1 = F.softmax(input1, dim=1)
        input2 = F.softmax(input2, dim=1)
        # 计算交集
        intersection = torch.sum(input1 * input2, dim=(2, 3))
        # 计算各自的总和
        input1_sum = torch.sum(input1, dim=(2, 3))
        input2_sum = torch.sum(input2, dim=(2, 3))
        # 计算 Dice 系数
        dice = (2. * intersection + smooth) / (input1_sum + input2_sum + smooth)
        # 计算 Dice Loss
        dice_loss = 1 - dice
        return {'consistency_loss': dice_loss.mean()*lamb}


    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    def forward_style(self, img, img_metas, **kwargs):
        return self.backbone.forward_features(img, return_style=True)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_semantic_seg,
                      seg_weight=None,
                      return_feat=False,
                      return_logits=False,
                      ):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        if self.diff_train:
            x_w_seg = self.extract_feat(img, gt_semantic_seg=gt_semantic_seg)
            x_wo_seg = self.extract_feat(img)

            losses = dict()
            if return_feat:
                losses['features'] = x_w_seg

            loss_decode_w_seg = self._decode_head_forward_train(x_w_seg, img_metas,
                                                        gt_semantic_seg,
                                                        seg_weight,
                                                        return_logits=True,
                                                        prefix='decode_w_seg')
            loss_decode_wo_seg = self._decode_head_forward_train(x_wo_seg, img_metas,
                                                        gt_semantic_seg,
                                                        seg_weight,
                                                        return_logits=True,
                                                        prefix='decode_wo_seg')
            
            logits_w_seg = loss_decode_w_seg['decode_w_seg.logits']
            logits_wo_seg = loss_decode_wo_seg['decode_wo_seg.logits']

            loss_consistency = self.mse_loss(logits_w_seg, logits_wo_seg, lamb=1.0)
            # loss_consistency = self.kl_loss(pred_logits=logits_wo_seg, ref_logits=logits_w_seg, lamb=0.1, temp_s=1, temp_t=1, max_v=0.5)
            
            # if self.iter > self.total_iter // 4:
            #     losses.update(loss_consistency)
            losses.update(loss_consistency)

            # print('jyxjyxjyx', loss_consistency['consistency_loss'], logits_w_seg.max(), logits_w_seg.min(), logits_wo_seg.max(), logits_wo_seg.min())

            # for k, v in loss_decode_w_seg.items():
            #     loss_decode_w_seg[k] = 0.5 * v
            # for k, v in loss_decode_wo_seg.items():
            #     loss_decode_wo_seg[k] = 0.5 * v
            
            losses.update(loss_decode_w_seg)
            losses.update(loss_decode_wo_seg)
            

        else:
            x = self.extract_feat(img)
            losses = dict()
            if return_feat:
                losses['features'] = x
            loss_decode_seg = self._decode_head_forward_train(x, img_metas,
                                                        gt_semantic_seg,
                                                        seg_weight,
                                                        return_logits)
            losses.update(loss_decode_seg)
        
        self.iter = min(self.iter + 1, self.total_iter)

        return losses

    # TODO refactor
    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batched_slide = self.test_cfg.get('batched_slide', False)
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        if batched_slide:
            crop_imgs, crops = [], []
            for h_idx in range(h_grids):
                for w_idx in range(w_grids):
                    y1 = h_idx * h_stride
                    x1 = w_idx * w_stride
                    y2 = min(y1 + h_crop, h_img)
                    x2 = min(x1 + w_crop, w_img)
                    y1 = max(y2 - h_crop, 0)
                    x1 = max(x2 - w_crop, 0)
                    crop_img = img[:, :, y1:y2, x1:x2]
                    crop_imgs.append(crop_img)
                    crops.append((y1, y2, x1, x2))
            crop_imgs = torch.cat(crop_imgs, dim=0)
            crop_seg_logits = self.encode_decode(crop_imgs, img_meta)
            for i in range(len(crops)):
                y1, y2, x1, x2 = crops[i]
                crop_seg_logit = \
                    crop_seg_logits[i * batch_size:(i + 1) * batch_size]
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        else:
            for h_idx in range(h_grids):
                for w_idx in range(w_grids):
                    y1 = h_idx * h_stride
                    x1 = w_idx * w_stride
                    y2 = min(y1 + h_crop, h_img)
                    x2 = min(x1 + w_crop, w_img)
                    y1 = max(y2 - h_crop, 0)
                    x1 = max(x2 - w_crop, 0)
                    crop_img = img[:, :, y1:y2, x1:x2]
                    crop_seg_logit = self.encode_decode(crop_img, img_meta)
                    preds += F.pad(crop_seg_logit,
                                   (int(x1), int(preds.shape[3] - x2), int(y1),
                                    int(preds.shape[2] - y2)))

                    count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds

    def whole_inference(self, img, img_meta, rescale, gt_semantic_seg=None):
        """Inference with full image."""

        seg_logit = self.encode_decode(img, img_meta, gt_semantic_seg=gt_semantic_seg)
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                size = img_meta[0]['ori_shape'][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return seg_logit

    def inference(self, img, img_meta, rescale, gt_semantic_seg=None):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale, gt_semantic_seg)
        if hasattr(self.decode_head, 'debug_output_attention') and \
                self.decode_head.debug_output_attention:
            output = seg_logit
        else:
            output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))

        return output

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)

        ########### For logit save
        # print('jyxjyxjyx logit save')
        # file_name = img_meta[0]['filename'].split('/')[-1].replace('_leftImg8bit.png', '.pt')
        # torch.save(seg_logit.cpu(), f'/home/xmuairmud/jyx/HRDA/save_attn/logit_ori/{file_name}')

        if hasattr(self.decode_head, 'debug_output_attention') and \
                self.decode_head.debug_output_attention:
            seg_pred = seg_logit[:, 0]
        else:
            seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred

        
        ############ For DIFT Mask
        # print('jyxjyxjyx', seg_pred.shape)
        # print(seg_pred)
        # seg_pred = seg_pred[:, None, :, :]
        # seg_logit = self.inference(img, img_meta, rescale, seg_pred)
        # seg_pred = seg_logit.argmax(dim=1)
        ############
        ############ For threshold Mask
        # threshold = 0.0
        # prob = torch.max(seg_logit, dim=1)[0]
        # # print('jyxjyxjyx', seg_logit.shape, seg_pred.shape)
        # pseudo_label = torch.where(prob > threshold, seg_pred, torch.full_like(seg_pred, 255))
        # print(f"prob: {prob.max()}")
        # # print(f"prob: {seg_logit.max()}")
        # print(f"psuedo_label: {(pseudo_label!=255).sum()}/{pseudo_label.shape[1]*pseudo_label.shape[2]}")
        # # print('jyxjyxjyx', pseudo_label.shape)
        # seg_logit = self.inference(img, img_meta, rescale, pseudo_label)
        # seg_pred = seg_logit.argmax(dim=1)
        ############

        seg_pred = seg_pred.cpu().numpy()

        ############# Visualization
        # print('jyxjyx visualization!!!')
        # colors = np.array([
        #     [128, 64,128],
        #     [244, 35,232],
        #     [ 70, 70, 70],
        #     [102,102,156],
        #     [190,153,153],
        #     [153,153,153],
        #     [250,170, 30],
        #     [220,220,  0],
        #     [107,142, 35],
        #     [152,251,152],
        #     [ 70,130,180],
        #     [220, 20, 60],
        #     [255,  0,  0],
        #     [  0,  0,142],
        #     [  0,  0, 70],
        #     [  0, 60,100],
        #     [  0, 80,100],
        #     [  0,  0,230],
        #     [119, 11, 32],
        # ])
        # vis_pred = seg_pred[0, ...]
        
        # color_image = colors[vis_pred]
        # color_image_pil = Image.fromarray(color_image.astype('uint8'), 'RGB')
        # file_name = img_meta[0]['filename'].split('/')[-1].replace('_rgb_anon', '')
        # # file_name = img_meta[0]['filename'].split('/')[-1].replace('_leftImg8bit', '')
        # # color_image_pil.save(f'/home/xmuairmud/data/mm2024/vis/DIFF/mv/mv_{file_name}')
        # color_image_pil.save(f'/home/xmuairmud/jyx/daily_scripts/18025_seg_wo_ref.png')
        # # file_name = img_meta[0]['filename'].split('/')[-1].replace('_leftImg8bit', '_predict')
        # # color_image_pil.save(f'/home/xmuairmud/jyx/HRDA/save_attn/logit_ori/{file_name}')
        # print(type(img_meta))

        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
