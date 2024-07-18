
from nodes import VAEDecode, VAEEncode
import comfy
from .utils import image_2dtransform, apply_easing, easing_functions
import torch
from tqdm import tqdm





class AstralAnimator:
 
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "vae": ("VAE", ),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "image": ("IMAGE", ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "frame": ("INT", {"default": 16}),
                "x": ("INT", {"default": 15, "step": 1, "min": -4096, "max": 4096}),
                "y": ("INT", {"default": 0, "step": 1, "min": -4096, "max": 4096}),
                "zoom": ("FLOAT", {"default": 0.98, "min": 0.001, "step": 0.01}),
                "angle": ("INT", {"default": 0, "step": 1, "min": -360, "max": 360}),
                "denoise_min": ("FLOAT", {"default": 0.40, "min": 0.00, "max": 1.00, "step":0.01}),
                "denoise_max": ("FLOAT", {"default": 0.60, "min": 0.00, "max": 1.00, "step":0.01}),
                "easing_type": (list(easing_functions.keys()), ),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("IMAGE", )
    FUNCTION = "animate"

    CATEGORY = "animation"

    def animate(self, model, vae, positive, negative, image, frame, seed, steps, cfg, sampler_name, scheduler, x, y, zoom, angle, denoise_min, denoise_max, easing_type):
        
        # Initialize the model
        vaedecode = VAEDecode()
        vaeencode = VAEEncode()

        res = [image]

        pbar = comfy.utils.ProgressBar(frame)
        for i in tqdm(range(frame)):
            
            # Calculating noise
            denoise = (denoise_max - denoise_min) * apply_easing((i+1)/frame, easing_type)  + denoise_min

            # Transform
            image = image_2dtransform(image, x, y, zoom, angle, 0, "reflect")

            # coding
            latent = vaeencode.encode(vae, image)[0]


            # Calculating noise
            noise = comfy.sample.prepare_noise(latent["samples"], i, None)

            # sampling
            samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent["samples"],
                                        denoise=denoise, disable_noise=False, start_step=None, last_step=None,
                                        force_full_denoise=False, noise_mask=None, callback=None, disable_pbar=True, seed=seed+i)
        
            # decoding
            image = vaedecode.decode(vae, {"samples": samples})[0]

            # You can also add preview pictures here
            pbar.update_absolute(i + 1, frame, None)

            res.append(image)

        # If the size of the first image and the generated image is inconsistent, it will be discarded
        if res[0].size() != res[-1].size():
            res = res[1:]

        res = torch.cat(res, dim=0)
        return (res, )
        
# Update NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS
NODE_CLASS_MAPPINGS = {
    "AstralAnimator": AstralAnimator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AstralAnimator": "Astral Animator"
}

# __init__.py
from .main import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
