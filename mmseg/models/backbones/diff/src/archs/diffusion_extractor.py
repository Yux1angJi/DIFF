# Extract Diffusion Features
# Implementation for DIFF for paper 'Diffusion Features to Bridge Domain Gap for Semantic Segmentation'
# Based on HyperFeature
# By Yuxiang Ji


from PIL import Image
import torch
from torch import nn

from diffusers import DDIMScheduler
from archs.stable_diffusion.diffusion import (
    init_models, 
    get_tokens_embedding,
    generalized_steps,
    collect_and_resize_feats,
    collect_stride_feats_with_timesteplist,
)
from archs.stable_diffusion.resnet import init_resnet_func, init_ca_resnet_func


class DiffusionExtractor(nn.Module):
    """
    Module for running either the generation or inversion process 
    and extracting intermediate feature maps.
    """
    def __init__(self, config, device):
        super().__init__()
        self.device = device

        self.pipe, self.unet, self.vae, self.clip, self.clip_tokenizer = init_models(model_id=config["model_id"])

        self.scheduler = self.pipe.scheduler

        self.num_timesteps = config["num_timesteps"]
        self.scheduler.set_timesteps(self.num_timesteps)
        self.scheduler.timesteps = torch.Tensor(config["scheduler_timesteps"])
        self.generator = torch.Generator(self.device).manual_seed(config.get("seed", 0))

        self.prompt = config.get("prompt", "")
        self.negative_prompt = config.get("negative_prompt", "")
        
        self.batch_size = 2
        self.mode = 'train'
        self.batch_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.cond = {i: None for i in self.batch_list}
        self.uncond = {i: None for i in self.batch_list}
        self.set_cond(self.prompt, self.negative_prompt)
        
        self.diffusion_mode = config.get("diffusion_mode", "generation")
        print(f"diffusion_extractor diffusion_mode: {self.diffusion_mode}")
        self.diffusion_version = config.get("model_id")
        print(f"diffusion_extractor diffusion version: {self.diffusion_version}")

        if "idxs_resnet" in config and config["idxs_resnet"] is not None:
            self.idxs_resnet = config["idxs_resnet"]
        else:
            self.idxs_resnet = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)]
        
        if "idxs_ca" in config and config["idxs_ca"] is not None:
            self.idxs_ca = config["idxs_ca"]
        else:
            self.idxs_ca = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)]

        print(f"diffusion extractor idxs_resnet={self.idxs_resnet}")
        print(f"diffusion extractor idxs_ca={self.idxs_ca}")

        self.output_resolution = (config["input_resolution"][0]//8, config["input_resolution"][1]//8)

        print(f"diffusion extractor len(timesteps)={self.scheduler.timesteps.shape}")
        print(f"diffusion extractor timesteps={self.scheduler.timesteps}")

        # Note that save_timestep is in terms of number of generation steps
        # save_timestep = 0 is noise, save_timestep = T is a clean image
        # generation saves as [0...T], inversion saves as [T...0]
        self.save_timestep = config.get("save_timestep", [])
        print(f"diffusion extractor save_timestep: {self.save_timestep}")

        self.s_tmin = config.get("s_tmin")
        self.s_tmax = config.get("s_tmax")
        print(f"diffusion extractor s_tmin={self.s_tmin}, s_tmax={self.s_tmax}")

        self.eta = config.get("eta", 0.0)
        print(f"diffusion extractor ddim eta={self.eta}")

        self.guidance_scale = config.get("guidance_scale", -1)
        print(f"diffusion guidance scale={self.guidance_scale}")
        
        self.label_map = config.get("label_map")
        print(f"diffusion mask label_map={self.label_map}")

        self.do_optim_steps = config.get("do_optim_steps", False)
        self.do_pndm_steps = config.get("do_pndm_steps", False)
        self.do_adpm_steps = config.get("do_adpm_steps", False)
        self.do_trand_steps = config.get("do_trand_steps", False)
        self.do_pndm_trand_steps = config.get("do_pndm_trand_steps", False)
        self.do_pndm_ddpm_steps = config.get("do_pndm_ddpm_steps", False)
        self.do_mask_steps = config.get("do_mask_steps", False)
        self.do_one_step = config.get("do_one_step", False)
        self.do_with_depth = config.get("do_with_depth", False)
        print(f"diffusion extractor do_optim_steps:{self.do_optim_steps}, \
              do_pndm_steps:{self.do_pndm_steps}, \
              do_adpm_steps:{self.do_adpm_steps}, \
              do_trand_steps:{self.do_trand_steps}, \
              do_pndm_trand_steps:{self.do_pndm_trand_steps}, \
              do_pndm_ddpm_steps:{self.do_pndm_ddpm_steps}, \
              do_mask_steps:{self.do_mask_steps}, \
              do_one_step:{self.do_one_step}, \
              do_with_depth:{self.do_with_depth}")

        if self.do_with_depth:
            self.depth_estimator = self.pipe.depth_estimator
            self.feature_extractor = self.pipe.depth_estimator
            for param in self.depth_estimator.parameters():
                param.requires_grad = False
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        else:
            self.depth_estimator = None
            self.feature_extractor = None
        
        self.gamma = None
        if self.do_adpm_steps:
            gamma_file = config.get("gamma_file")
            self.gamma = torch.load(gamma_file).to(self.unet.device)
            print(f"Load Gamma file from {gamma_file}")

        if self.do_trand_steps:
            self.tclass_list = config.get("tclass_list")
        else:
            self.tclass_list = None
        
        self.mask_min = config.get("mask_min", 0)
        self.mask_max = config.get("mask_max", 1000)
        if self.do_mask_steps:
            print(f"do_mask_steps with mask_min={self.mask_min}, mask_max={self.mask_max}")

        # self.blip_model = BlipForConditionalGeneration.from_pretrained(config.get("blip_model_id"), torch_dtype=torch.float16).to("cuda")

        # print(f"batch_size: {self.batch_size}")
        # print(f"diffusion_mode: {self.diffusion_mode}")
        # print(f"idxs: {self.idxs}")
        # print(f"output_resolution: {self.output_resolution}")
        # print(f"prompt: {self.prompt}")
        # print(f"negative_prompt: {self.snegative_prompt}")

    def set_cond(self, prompt, negative_prompt):
        print('prompt', prompt)
        print('negative_prmopt', negative_prompt)
        with torch.no_grad():
                with torch.autocast("cuda"):
                    _, cond_prompt = get_tokens_embedding(self.clip_tokenizer, self.clip, self.device, prompt)
                    _, uncond_prompt = get_tokens_embedding(self.clip_tokenizer, self.clip, self.device, negative_prompt)
        for batch_size in self.batch_list:
            cond_tmp = cond_prompt.expand((batch_size, *cond_prompt.shape[1:]))
            cond_tmp = cond_tmp.to(self.device)

            uncond_tmp = uncond_prompt.expand((batch_size, *uncond_prompt.shape[1:]))
            uncond_tmp = uncond_tmp.to(self.device)

            self.cond[batch_size] = cond_tmp
            self.uncond[batch_size] = uncond_tmp

    def change_mode(self, mode='val'):
        self.mode = mode
    
    def change_batchsize(self, batch_size):
        self.batch_size = batch_size

    def change_cond(self, prompt, cond_type="cond", batch_size=2):
        with torch.no_grad():
            with torch.autocast("cuda"):
                _, new_cond = get_tokens_embedding(self.clip_tokenizer, self.clip, self.device, prompt)
                new_cond = new_cond.expand((batch_size, *new_cond.shape[1:]))
                new_cond = new_cond.to(self.device)
                if cond_type == "cond":
                    self.cond = new_cond
                    self.prompt = prompt
                elif cond_type == "uncond":
                    self.uncond = new_cond
                    self.negative_prompt = prompt
                else:
                    raise NotImplementedError
                
    def to_image(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        images = self.vae.decode(latents).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        # print(image.shape)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        images = (images * 255).round().astype("uint8")
        return [Image.fromarray(image) for image in images]

    def run_generation(self, latent, guidance_scale=-1, min_i=None, max_i=None, 
                       mode="norm", sag_scale=0.3, do_pndm_steps=True):
        if mode == "norm":
            xs = generalized_steps(
                latent,
                self.unet,
                self.scheduler,
                run_inversion=False, 
                guidance_scale=guidance_scale,
                do_pndm_steps=do_pndm_steps,
                conditional=self.cond[self.batch_size],
                unconditional=self.uncond[self.batch_size],
                min_i=min_i,
                max_i=max_i
            )
            # images = self.to_image(xs)
            # images[0].save('/home/xmuairmud/jyx/daily_scripts/inv_rst_pndmh_test.png')
        elif mode == "sag":
            xs = generalized_steps_sag(
                latent,
                self.unet, 
                self.scheduler, 
                run_inversion=False, 
                guidance_scale=guidance_scale, 
                sag_scale=sag_scale,
                do_pndm_steps=do_pndm_steps,
                conditional=self.cond[self.batch_size], 
                unconditional=self.uncond[self.batch_size], 
                min_i=min_i,
                max_i=max_i
            )
            # images = self.to_image(xs)
            # images[0].save('/home/xmuairmud/jyx/daily_scripts/inv_rst_sagf_05_pndm_test.png')
        return xs

    def run_inversion(self, latent, gt_semantic_seg=None, images=None, min_i=None, max_i=None, 
                      mode="norm", sag_scale=0.5):
        if mode == "norm":
            xs = generalized_steps(
                latent, 
                self.unet, 
                self.scheduler,
                feature_extractor=self.feature_extractor,
                depth_estimator=self.depth_estimator,
                images=images,
                gt_semantic_seg=gt_semantic_seg,
                tclass_list=self.tclass_list,
                run_inversion=True, 
                guidance_scale=self.guidance_scale, 
                do_pndm_steps=self.do_pndm_steps,
                do_optim_steps=self.do_optim_steps,
                do_adpm_steps=self.do_adpm_steps,
                do_trand_steps=self.do_trand_steps,
                do_pndm_trand_steps=self.do_pndm_trand_steps,
                do_pndm_ddpm_steps=self.do_pndm_ddpm_steps,
                do_mask_steps=self.do_mask_steps,
                do_one_step=self.do_one_step,
                do_with_depth=self.do_with_depth,
                label_map=self.label_map,
                gamma=self.gamma,
                prompt=self.prompt,
                negative_prompt=self.negative_prompt,
                conditional=self.cond[self.batch_size],
                unconditional=self.uncond[self.batch_size],
                min_i=min_i,
                max_i=max_i,
                s_tmin=self.s_tmin,
                s_tmax=self.s_tmax,
                mask_min=self.mask_min,
                mask_max=self.mask_max,
                eta=self.eta,
                clip=self.clip,
                clip_tokenizer=self.clip_tokenizer,
            )
        elif mode == "sag":
            xs = generalized_steps_sag(
                latent, 
                self.unet, 
                self.scheduler, 
                run_inversion=True, 
                guidance_scale=self.guidance_scale,
                sag_scale=sag_scale,
                do_pndm_steps=self.do_pndm_steps,
                conditional=self.cond[self.batch_size],
                unconditional=self.uncond[self.batch_size],
                min_i=min_i,
                max_i=max_i
            )
        return xs

    def get_feats(self, latents, extractor_fn, preview_mode=False):
        # returns feats of shape [batch_size, num_timesteps, channels, w, h]
        if not preview_mode:
            init_resnet_func(self.unet, save_hidden=True, reset=True, idxs=self.idxs, save_timestep=self.save_timestep)
        outputs = extractor_fn(latents)
        if not preview_mode:
            feats = []
            for timestep in self.save_timestep:
                timestep_feats = collect_and_resize_feats(self.unet, self.idxs, timestep, self.output_resolution)
                feats.append(timestep_feats)
            feats = torch.stack(feats, dim=1)
            init_resnet_func(self.unet, reset=True)
        else:
            feats = None
        return feats, outputs
    
    def get_feats_stride(self, latents, extractor_fn, preview_mode=False, do_optim_steps=False):
        # returns feats of shape [batch_size, num_timesteps, channels, w, h]
        if not preview_mode:
            init_ca_resnet_func(self.unet, save_hidden=True, reset=True, idxs_resnet=self.idxs_resnet, idxs_ca=self.idxs_ca, save_timestep=self.save_timestep)
        outputs = extractor_fn(latents)
        if not preview_mode:
            feats = collect_stride_feats_with_timesteplist(self.unet, self.idxs_resnet, self.idxs_ca, timestep_list=self.save_timestep, 
                                                           do_mask_steps=self.do_mask_steps, x=latents)
            # feats = torch.stack(feats, dim=1)
            init_ca_resnet_func(self.unet, reset=True)
        else:
            feats = None
        return feats
    
    def get_feats_stride_inv_rst(self, latents):
        latents_inv = self.run_inversion(latents, mode='norm')

        init_resnet_func(self.unet, save_hidden=True, reset=True, idxs=self.idxs, save_timestep=self.save_timestep)
        self.run_generation(latents_inv, mode='norm')
        feats = collect_stride_feats_with_timesteplist(self.unet, self.idxs, timestep_list=self.save_timestep)
        init_resnet_func(self.unet, reset=True)

        return feats

    def latents_to_images(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        images = self.vae.decode(latents.to(self.vae.dtype)).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype("uint8")
        return [Image.fromarray(image) for image in images]

    def forward(self, images=None, latents=None, guidance_scale=-1, preview_mode=False, stride_mode=False, gt_semantic_seg=None):
        if images is None:
            if latents is None:
                latents = torch.randn((self.batch_size, self.unet.in_channels, 512 // 8, 512 // 8), device=self.device, generator=self.generator)
            if self.diffusion_mode == "generation":
                if preview_mode:
                    extractor_fn = lambda latents: self.run_generation(latents, guidance_scale, max_i=self.end_timestep)
            elif self.diffusion_mode == "inversion":
                raise NotImplementedError
        else:
            # images = torch.nn.functional.interpolate(images, size=512, mode="bilinear")
            latents = self.vae.encode(images).latent_dist.sample(generator=None) * 0.18215

            # Run BLIP
            # caption_prompt = self.blip_model.generate({"image": images})
            # batch_size_tmp = images.shape[0]
            # self.change_cond(caption_prompt, cond_type="cond", batch_size=batch_size_tmp)

            # print('jyxjyxjyx vae latents', torch.isnan(latents).float().sum())
            if self.diffusion_mode == "inversion":
                extractor_fn = lambda latents: self.run_inversion(latents, gt_semantic_seg=gt_semantic_seg, images=images)
            elif self.diffusion_mode == "inversion_sag":
                extractor_fn = lambda latents: self.run_inversion(latents, mode='sag')
            elif self.diffusion_mode == "generation":
                raise NotImplementedError
        
        with torch.no_grad():
            with torch.autocast("cuda"):
                if self.diffusion_mode == "inv_then_rst":
                    return self.get_feats_stride_inv_rst(latents)
                else:
                    if stride_mode:
                        return self.get_feats_stride(latents, extractor_fn)
                    else:
                        return self.get_feats(latents, extractor_fn, preview_mode=preview_mode)
                