import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--clip_skip", type=int, default=1)
    parser.add_argument("--size", type=int, default=512)
    return parser.parse_args()


args = parse_args()
safetensor_path = args.model_path
clipskip = args.clip_skip
size = args.size
resolution = (size, size)

# Please use at least 10 prompts, and the prompts should preferably use words related to the model's usage scenarios.
prompts = [
    [
        "masterpiece,best quality,1girl,solo,long hair,looking at viewer,red eyes,bangs,shirt,blush,collarbone,white shirt,upper body,smile,closed mouth,hair between eyes,white hair,hair ornament,",
        "low quality,blurry,lowres,worstquality,normal quality,bad proportions,out of focus,logo,text,twisted limbs,",
    ],
    [
        "1girl,animal ears,tail,cat ears,green eyes,solo,cat tail,thighhighs,hair ornament,short hair,brown hair,bell,dress,fang,heart hands,open mouth,wings,cat girl,full body,ribbon,twintails,",
        "bad anatomy,worst quality,low quality,blurry,monochrome,comic,greyscale,text,watermark,multiple views,pubic hair,sketches,fused fingers,fused toes,bad feet,missing toe,extra toes,missing limb,missing finger,extra fingers,ugly hands,deformed hands,fused hands,fused eyes,liquid eyes,fused pupils,liquid pupils,censored,mosaic,minigirl,short hair,very short hair,EasyNegativeV2,AnimeSupport-neg,badhandv4,",
    ],
    [
        "1girl,bangs,beret,blush,braid,dress,eyebrows visible through hair,full body,hair between eyes,hair ornament,hairclip,hat,long hair,long sleeves,looking at viewer,parted lips,pink hair,red headwear,shoes,solo,standing,sweater,twin braids,very long hair,white legwear,x hair ornament,outdoors,",
        "2boy,lowres,bad anatomy,bad hands,text,error,missing fingers,extra digit,fewer digits,worst quality,low quality,normal quality,jpeg artifacts,signature,watermark,username,blurry,lowres graffiti,low quality lowres simple background,multiple people,Multiple people,bad body,ugly,poorly drawn face,malformed hands,poorly drawn hands,mutated hands,too many fingers,mutated hands and fingers,extra legs,extra limb,extra arms,disconnected limbs,monochrome,bad proportions,worstquality,grayscale,long body,EasyNegativeV2,",
    ],
    [
        "solo,1girl,looking at viewer,Noelle,blush,full body,black_shoes,blush,laughing,hand up,teasing_smile,kind_smile,white gloves,white dress,bathtub,wet,",
        "badhandv4,BadNegAnatomyV1-neg,Multiple people,bad hands,malformed hands,mutated hands,missing fingers,fused fingers,floating limbs,malformed limbs,low quality,bad anatomy,lowres,sweat,bad proportions,poorly drawn hands,too many fingers,mutated hands and fingers,text,",
    ],
    [
        "1girl,lingjia,pink hair,blue eyes,big eyes,uniforms,standing,hat,round face,kawaii,round face, masterpiece, best quality,",
        "badhandv4,EasyNegativeV2,ng_deepnegative_v1_75t,EasyNegative,, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
    ],
    [
        "Satou, happy, blushing, distant view, portrait, , raining, outside, wet clothes, emotionless, expressionless, dreary atmosphere, sad atmosphere",
        "easynegative, closed eyes, expressionless, emotionless, angry, sad, indoors, school, apartment, side view, rear view,  bad hands, weird hands,",
    ],
    [
        "unparalleled masterpiece, masterpiece, best quality, high quality, absurdres, 1girl, maniwa roka, wa maid, pinstripe kimono, kimono, apron, hair bow, striped kimono, long hair, hair between eyes, frilled apron, frilled sleeves, mole under eye, aqua eyes",
        "easynegative, bad-hands-5, extra fingers, fewer fingers, low quality, worst quality, bad anatomy, inaccurate limb, bad composition, inaccurate eyes, extra digit,fewer digits, extra arms, medium quality, deleted, lowres, comic, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, jpeg artifacts, signature, watermark, username, blurry, negative space, red headband, less fingers, deformed hands, polydactyl, chromatic abberation, deformed limbs, extra limbs, fewer fingers, extra fingers, crooked fingers, loli, shota, twintails, cleavage cutout",
    ],
    [
        "masterpiece, best quality, ultra-detailed, realistic, 8k, one pine tree, flowers fields, many kinds of flowers, a couple of rabbits on very distant, grass, glitter, midnight, dark vibe, dark light, dim light, rim light, stars, natural night light, fruits trees, big fruits trees, a lots of fruits trees, out-of-this-world scenery, lifelike, naturalism, greatest naturalism photo, immaculate detailed realism photography, rich colors, best colors perfect composition, perfect colors, colorful, clear photo, intrigue nature panorama, delicate nature photo, immaculate micro-details nature elements, out-of-this-world scenery, landscape, flowers, crystal fantastic, breathtaking beautiful nature, unmatched beauty nature, more nature elements, richest nature elements, aesthetic, artistic style, surreal, insanely lifelike, top best naturalism photography, greatest detailed, intricate details, depth of field, no human, detailerlora, a lots kinds of flowers, JZCG001,",
        "negative_hand, NegfeetV2, Realisian-Neg, worst quality, low quality, normal quality, poorly drawn, lowres, low resolution, signature, watermarks, ugly, out of focus, error, blurry, unclear photo, bad photo, unrealistic, semi realistic, pixelated, cartoon, anime, cgi, drawing, bra, panties, dress, shirt, clothes, small breast, medium breast, extra limbs, moon, milky way, 2d, 3d, censored,",
    ],
    [
        "Photorealism, cinematic film still, a woman in a dress is dancing in the living room with an antique radio in the background, Elizabeth Polunin, cinematic photography, a photorealistic painting, arabesque, Fujifilm Superia Venus 800 rendered in unreal engine, photorealistic",
        "anime, cartoon, muscular, simplified, unrealistic, impressionistic, low resolution, earthly, mundane, simple, simplified, abstract, unrealistic, impressionist, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured, figuration-libre, easynegative",
    ],
    [
        "masterpiece, best quality, ultra-detailed, realistic, 8k, capture the most beautiness nature in a glowing morning dew crystal, a pine tree, flowers fields, many kinds of flowers, a couple of rabbits on very distant, lakes, meadows, glitter, moonlight, out-of-this-world scenery, lifelike, naturalism, greatest naturalism photo, immaculate detailed realism photography, rich colors, best colors perfect composition, perfect colors, colorful, clear photo, intrigue nature panorama, delicate nature photo, immaculate micro-details nature elements, out-of-this-world scenery, landscape, flowers, crystal fantastic, breathtaking beautiful nature, unmatched beauty nature, more nature elements, rich nature elements, aesthetic, artistic style, surreal, insanely lifelike, top best naturalism photography, greatest detailed, intricate details, depth of field, no human, detailerlora,",
        "negative_hand, NegfeetV2, Realisian-Neg, worst quality, low quality, normal quality, poorly drawn, lowres, low resolution, signature, watermarks, ugly, out of focus, error, blurry, unclear photo, bad photo, unrealistic, semi realistic, pixelated, cartoon, anime, cgi, drawing, bra, panties, dress, shirt, clothes, small breast, medium breast, extra limbs, 2d, 3d, censored, girl, human, woman, human,",
    ],
]

import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline
from transformers import CLIPTokenizer
from tqdm import tqdm
import pickle as pkl
from diffusers import (
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
)
import os

data = {"clip": [], "unet": [], "vae": []}

pipe = StableDiffusionPipeline.from_single_file(
    safetensor_path,
    safety_checker=None,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
if clipskip > 1:
    pipe.text_encoder.text_model.encoder.layers = (
        pipe.text_encoder.text_model.encoder.layers[: -(clipskip - 1)]
    )
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)

pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config,
    algorithm_type="dpmsolver++",
    solver_order=2,
    solver_type="midpoint",
    lower_order_final=True,
    use_karras_sigmas=True,
    beta_schedule="scaled_linear",
)

text_encoder = pipe.text_encoder
unet = pipe.unet
vae = pipe.vae
tokenizer = pipe.tokenizer
scheduler = pipe.scheduler

idx = 0


def generate(prompt, negative_prompt, cfg, steps):
    global idx
    idx += 1

    width, height = resolution
    num_inference_steps = steps
    guidance_scale = cfg
    num_images_per_prompt = 1
    do_classifier_free_guidance = guidance_scale > 0

    text_inputs = tokenizer(
        [negative_prompt, prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids.to(device)

    data["clip"].append([text_input_ids.cpu().numpy()])

    with torch.no_grad():
        prompt_embeds = text_encoder(text_input_ids)[0]

    num_channels_latents = unet.config.in_channels
    latents_shape = (
        num_images_per_prompt,
        num_channels_latents,
        height // 8,
        width // 8,
    )

    latents = torch.randn(latents_shape, device=device, dtype=prompt_embeds.dtype)

    scheduler.set_timesteps(num_inference_steps, device=device)
    latents = latents * scheduler.init_noise_sigma

    for i, t in tqdm(enumerate(scheduler.timesteps)):
        latent_model_input = (
            torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        )
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        data["unet"].append(
            [
                latent_model_input.cpu().numpy(),
                t.cpu().numpy() if hasattr(t, "cpu") else np.array([t]),
                prompt_embeds.cpu().numpy(),
            ]
        )
        with torch.no_grad():
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
            ).sample

        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
        scheduler_output = scheduler.step(noise_pred, t, latents)
        latents = scheduler_output.prev_sample

    latents = 1 / vae.config.scaling_factor * latents

    data["vae"].append([latents.cpu().numpy()])

    with torch.no_grad():
        image = vae.decode(latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image * 255).round().astype("uint8")
    image = image[0]

    pil_image = Image.fromarray(image)

    os.makedirs("images", exist_ok=True)
    pil_image.save(f"images/{idx}.png")
    print(f"image saved images/{idx}.png")


for prompt, negative_prompt in prompts:
    for cfg in [7]:
        print(prompt, negative_prompt, cfg, 28 - idx % 9)
        generate(prompt, negative_prompt, cfg, 28 - idx % 9)

with open("data.pkl", "wb") as f:
    pkl.dump(data, f)
