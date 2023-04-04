import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
import pathlib
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
from einops import rearrange, repeat

from src.ldm.util import instantiate_from_config
from src.ldm.models.diffusion.ddim import DDIMSampler
from src.ldm.models.diffusion.plms import PLMSSampler

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor


# load safety model
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model

def load_img(path, h0, w0):

    image = Image.open(path).convert("RGB")
    w, h = image.size

    if h0 is not None and w0 is not None:
        h, w = h0, w0
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0

def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    has_nsfw_concept = [i if i == False else not i for i in has_nsfw_concept]
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept

class SDBot:
    def __init__(self):
        self.config = OmegaConf.load("src/configs/stable-diffusion/v1-inference.yaml")
        # self.model = load_model_from_config(self.config , "src/models/ldm/stable-diffusion-v1/mdjrny-v4.ckpt")
        self.model = load_model_from_config(self.config , "src/models/ldm/stable-diffusion-v1/Mdjrny-pprct_step_7000.ckpt")
        # self.model_tier = load_model_from_config(self.config , "src/models/ldm/stable-diffusion-v1/Mdjrny-pprct_step_7000.ckpt")
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = self.model.to(self.device)
        # self.model_tier = self.model_tier.to(self.device)
        # self.sampler = PLMSSampler(self.model) ## assume --plms by default
        self.sampler = DDIMSampler(self.model) ## assume --plms by default
        self.n_samples = 1
        self.output_path = '' ## TODO
        self.precision = 'autocast'
        self.precision_scope = autocast if self.precision=="autocast" else nullcontext
        self.start_code = None
        # if self.fixed_code:
        #     self.start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)
        self.strength = 0.75
        self.W = 512
        self.H = 512
        self.ddim_steps = 50
        self.scale = 7.5
        self.ddim_eta = 0
        self.n_iter = 1
        self.C = 4
        self.f = 8
        self.n_samples = 1
        self.batch_size = self.n_samples
        self.outpath = './outputs'
        self.sample_path = os.path.join(self.outpath, "samples")
        self.img_path = './outputs'
        os.makedirs(self.sample_path, exist_ok=True)
        self.base_count = len(os.listdir(self.sample_path))

    def img2img(self, prompt, filename, seed=None, negative_prompt = ""):

        if seed != None:
            seed_everything(seed)

        tier = ""
        init_image_path = ""

        if "--" in prompt:
            prompt_arr = prompt.split("--")
            tier = prompt_arr[1].strip()
            prompt = prompt_arr[0]

        if tier == "Bronze":
            init_image_path = "./tier_images/bronze.png"
            prompt = "mdjrny-pprct " + prompt
        if tier == "Silver":
            init_image_path = "./tier_images/silver.png"
            prompt = "mdjrny-pprct " + prompt
        if tier == "Gold":
            init_image_path = "./tier_images/gold.png"
            prompt = "mdjrny-pprct " + prompt

        init_image = load_img(init_image_path, 512, 512).to(self.device)
        init_image = repeat(init_image, "1 ... -> b ...", b=self.batch_size)
        init_latent = self.model.get_first_stage_encoding(self.model.encode_first_stage(init_image))
        self.data = [self.batch_size * [prompt]]
        self.sampler.make_schedule(ddim_num_steps=self.ddim_steps, ddim_eta=self.ddim_eta, verbose=False)
        
        with torch.no_grad():
            with self.precision_scope("cuda"):
                with self.model.ema_scope():
                    all_samples = list()
                    for n in trange(self.n_iter, desc="Sampling"):
                        for prompts in tqdm(self.data, desc="data"):
                            uc = None
                            if negative_prompt != "":
                                uc = self.model.get_learned_conditioning(self.batch_size * [negative_prompt])
                            if self.scale != 1.0:
                                uc = self.model.get_learned_conditioning(self.batch_size * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = self.model.get_learned_conditioning(prompts)
                            z_enc = self.sampler.stochastic_encode(
                                    init_latent,
                                    torch.tensor([int(self.strength * self.ddim_steps)] * self.batch_size).to(self.device)
                                )
                            samples_ddim = self.sampler.decode(
                                z_enc,
                                c,
                                int(self.strength * self.ddim_steps),
                                unconditional_guidance_scale=self.scale,
                                unconditional_conditioning=uc,
                            )

                            x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                            x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                            x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)
                            x_checked_image = x_samples_ddim
                            x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                            
                            for x_sample in x_checked_image_torch:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                img.save(os.path.join(self.img_path, tier + '_' + filename + '.png'))
                                self.base_count += 1


    def makeimg(self, prompt, filename, seed=None, negative_prompt = ""):
        if seed != None:
            seed_everything(seed)

        tier = ""

        if "--" in prompt:
            prompt_arr = prompt.split("--")
            tier = prompt_arr[1]
            prompt = prompt_arr[0]

        if tier == "Bronze":
            prompt = "mdjrny-pprct, midjourney paper cut " + prompt
            filename = tier + '_' + filename
        if tier == "Silver":
            prompt = "mdjrny-pprct, midjourney paper cut " + prompt
            filename = tier + '_' + filename
        if tier == "Gold":
            prompt = "mdjrny-pprct, midjourney paper cut " + prompt
            filename = tier + '_' + filename

        self.data = [prompt]  #one prompt at a time

        # if tier == "":
        if True:
            with torch.no_grad():
                with self.precision_scope("cuda"):
                    with self.model.ema_scope():
                        for n in trange(self.n_iter, desc="Sampling"):
                            for prompts in tqdm(self.data, desc="data"):
                                uc = None
                                if negative_prompt != "":
                                    uc = self.model.get_learned_conditioning(self.batch_size * [negative_prompt])
                                if self.scale != 1.0:
                                    uc = self.model.get_learned_conditioning(self.batch_size * [""])
                                if isinstance(prompts, tuple):
                                    prompts = list(prompts)
                                c = self.model.get_learned_conditioning(prompts)
                                shape = [self.C, self.H // self.f, self.W // self.f]
                                samples_ddim, _ = self.sampler.sample(S=self.ddim_steps,
                                                                conditioning=c,
                                                                batch_size=self.n_samples,
                                                                shape=shape,
                                                                verbose=False,
                                                                unconditional_guidance_scale=self.scale,
                                                                unconditional_conditioning=uc,
                                                                eta=self.ddim_eta,
                                                                x_T=self.start_code)

                                x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                                x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                                #x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)
                                x_checked_image = x_samples_ddim
                                x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                                
                                for x_sample in x_checked_image_torch:
                                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                    img = Image.fromarray(x_sample.astype(np.uint8))
                                    #img = put_watermark(img, wm_encoder)
                                    img.save(os.path.join(self.img_path, filename + '.png'))
                                    self.base_count += 1

                        toc = time.time()
        # else :
            # with torch.no_grad():
            #     with self.precision_scope("cuda"):
            #         with self.model_tier.ema_scope():
            #             for n in trange(self.n_iter, desc="Sampling"):
            #                 for prompts in tqdm(self.data, desc="data"):
            #                     uc = None
            #                     if negative_prompt != "":
            #                         uc = self.model_tier.get_learned_conditioning(self.batch_size * [negative_prompt])
            #                     if self.scale != 1.0:
            #                         uc = self.model_tier.get_learned_conditioning(self.batch_size * [""])
            #                     if isinstance(prompts, tuple):
            #                         prompts = list(prompts)
            #                     c = self.model_tier.get_learned_conditioning(prompts)
            #                     shape = [self.C, self.H // self.f, self.W // self.f]
            #                     samples_ddim, _ = self.sampler.sample(S=self.ddim_steps,
            #                                                     conditioning=c,
            #                                                     batch_size=self.n_samples,
            #                                                     shape=shape,
            #                                                     verbose=False,
            #                                                     unconditional_guidance_scale=self.scale,
            #                                                     unconditional_conditioning=uc,
            #                                                     eta=self.ddim_eta,
            #                                                     x_T=self.start_code)

            #                     x_samples_ddim = self.model_tier.decode_first_stage(samples_ddim)
            #                     x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            #                     x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

            #                     #x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)
            #                     x_checked_image = x_samples_ddim
            #                     x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                                
            #                     for x_sample in x_checked_image_torch:
            #                         x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
            #                         img = Image.fromarray(x_sample.astype(np.uint8))
            #                         #img = put_watermark(img, wm_encoder)
            #                         img.save(os.path.join(self.img_path, tier + '_' + filename + '.png'))
            #                         self.base_count += 1

            #             toc = time.time()
                        
