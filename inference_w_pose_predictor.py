import os
import argparse
import torch
from PIL import Image
from torchvision import transforms
from omegaconf import OmegaConf
from diffusers import UNet2DModel, UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from pipelines.Fashion_MAC_pipeline_pose import Fashion_MAC_pose
from pipelines.Fashion_MAC_pipeline import Fashion_MAC
from adapter.attention_processor import CAttnProcessor2_0_RADA, SAttnProcessor2_0
from adapter.modules.autoencoder import VQModel

def load_vqvae_model(config_path, ckpt_path):
    config = OmegaConf.load(config_path)
    print(f"Loading VQModel from {config_path}...")
    model = VQModel(**config.model.params)
    print(f"Loading weights from {ckpt_path}...")
    pl_sd = torch.load(ckpt_path, map_location="cpu")
    sd = pl_sd["state_dict"] if "state_dict" in pl_sd else pl_sd
    keys_to_ignore = [k for k in sd.keys() if k.startswith("loss") or k.startswith("monitor")]
    for k in keys_to_ignore:
        del sd[k]
    model.load_state_dict(sd, strict=False)
    return model

def modify_unet_input_channels(original_unet, new_in_channels=13):
    original_conv = original_unet.conv_in
    original_weight = original_conv.weight.data 
    original_bias = original_conv.bias.data if original_conv.bias is not None else None
    
    new_conv = torch.nn.Conv2d(
        new_in_channels,
        original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=original_conv.bias is not None
    )
    with torch.no_grad():
        new_conv.weight[:, :4] = original_weight.to(new_conv.weight.dtype) 
        if new_in_channels > 4:
            torch.nn.init.normal_(new_conv.weight[:, 4:], mean=0, std=0.02)
        if original_bias is not None:
            new_conv.bias.data = original_bias.to(new_conv.weight.dtype)
            
    original_unet.conv_in = new_conv
    return original_unet

def setup_pose_pipeline(args, vae_clo):
    vae_pose = load_vqvae_model(args.vae_pose_config_path, args.vae_pose_path)
    vae_pose.eval().to(device=args.device)

    model_state_dict = torch.load(args.pose_unet_path, map_location="cpu")["module"]
    unet_pose = UNet2DModel(
        sample_size=(args.height//8, args.width//8), 
        in_channels=7, 
        out_channels=3, 
        block_out_channels=[320, 640, 1280, 1280]
    )
    unet_pose.load_state_dict(model_state_dict, strict=True)

    noise_scheduler_pose = DDIMScheduler(
        num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, 
        beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False, steps_offset=1,
    )

    pipe_pose = Fashion_MAC_pose(
        vae_clo=vae_clo, vae_pose=vae_pose, unet=unet_pose, scheduler=noise_scheduler_pose
    ).to(dtype=torch.float16, device=args.device)
    return pipe_pose

def setup_tryon_pipeline(args, vae):
    if not os.path.exists(args.model_ckpt):
        raise ValueError(f"Tryon Checkpoint does not exist: {args.model_ckpt}")
    
    model_state_dict = torch.load(args.model_ckpt, map_location="cpu")["module"]
    unet_tryon = UNet2DConditionModel.from_pretrained(args.unet_pretrained_path, subfolder="unet").to(dtype=torch.float16, device=args.device)
    unet_tryon = modify_unet_input_channels(unet_tryon, new_in_channels=13)
    
    attn_procs = {}
    for name in unet_tryon.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet_tryon.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet_tryon.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet_tryon.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet_tryon.config.block_out_channels[block_id]
        
        if cross_attention_dim is not None:
            attn_procs[name] = CAttnProcessor2_0_RADA(
                name, hidden_size=hidden_size, h=768, w=576, 
                context_num=args.context_num, cross_attention_dim=cross_attention_dim,
            )
        else:
            attn_procs[name] = SAttnProcessor2_0(name, hidden_size)
    
    unet_tryon.set_attn_processor(attn_procs)
    unet_tryon.load_state_dict(model_state_dict, strict=True)
    unet_tryon.to(dtype=torch.float16, device=args.device)

    tokenizer = CLIPTokenizer.from_pretrained(args.unet_pretrained_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.unet_pretrained_path, subfolder="text_encoder").to(dtype=torch.float16, device=args.device)
    feature_extractor = CLIPImageProcessor.from_pretrained(args.unet_pretrained_path, subfolder="feature_extractor")

    noise_scheduler_tryon = DDIMScheduler(
        num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, 
        beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False, steps_offset=1
    )

    pipe_tryon = Fashion_MAC(
        unet=unet_tryon, vae=vae, tokenizer=tokenizer, context_num=args.context_num,
        text_encoder=text_encoder, scheduler=noise_scheduler_tryon,
        safety_checker=StableDiffusionSafetyChecker, feature_extractor=feature_extractor
    )
    return pipe_tryon

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def prepare(args):
    vae = AutoencoderKL.from_pretrained(args.vae_path).to(dtype=torch.float16, device=args.device)
    print("Loading Pose Generation Pipeline...")
    pipe_pose = setup_pose_pipeline(args, vae_clo=vae)
    print("Loading Virtual Try-on Pipeline...")
    pipe_tryon = setup_tryon_pipeline(args, vae)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    return pipe_pose, pipe_tryon, generator

def process_generated_densepose(pil_image, device, dtype=torch.float16):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    tensor = transform(pil_image).unsqueeze(0)
    tensor = torch.cat([tensor, torch.zeros_like(tensor)], dim = -1)
    return tensor.to(device=device, dtype=dtype)

def load_and_process_image(image_path, height, width, is_mask=False, device="cuda", dtype=torch.float16):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    img = Image.open(image_path)
    img = img.resize((width, height), Image.Resampling.BILINEAR if not is_mask else Image.Resampling.NEAREST)

    if is_mask:
        img = img.convert("L")
        transform = transforms.Compose([transforms.ToTensor()]) # [0, 1]
    else:
        img = img.convert("RGB")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) # [-1, 1]
        ])
    tensor = transform(img).unsqueeze(0) # [1, C, H, W]
    if is_mask:
        tensor = (tensor > 0.5).float()
    return tensor.to(device=device, dtype=dtype)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FashionMAC Inference with Pose Predictor')
    
    parser.add_argument('--model_ckpt', type=str, default="ckpt/FashionMAC_unet.pt")
    parser.add_argument('--unet_pretrained_path', type=str, default="ckpt/stable-diffusion-v1-4")
    parser.add_argument('--vae_path', type=str, default="ckpt/sd-vae-ft-ema/")
    parser.add_argument('--pose_unet_path', type=str, default="ckpt/pose_predictor/pose_predictor_unet.pt")
    parser.add_argument('--vae_pose_config_path', type=str, default="ckpt/pose_predictor/vqvae_config.yaml")
    parser.add_argument('--vae_pose_path', type=str, default="ckpt/pose_predictor/vqvae_f8_step=58239.ckpt")
    
    parser.add_argument('--output_path', type=str, default="inference_result")
    parser.add_argument('--device', type=str, default="cuda:1")
    parser.add_argument('--height', type=int, default=768)
    parser.add_argument('--width', type=int, default=576)
    parser.add_argument('--context_num', type=int, default=9)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--guidance_scale', type=float, default=2.5)
    parser.add_argument('--use_face', type=str2bool, default=True)

    parser.add_argument('--cloth_path', type=str, required=True, help="Path to the clothing image")
    parser.add_argument('--cloth_mask_path', type=str, required=True, help="Path to the clothing mask")
    parser.add_argument('--face_path', type=str, required=True, help="Path to the face image")
    
    parser.add_argument('--image_prompt', type=str, default="A model wearing a shirt", help="Global image prompt")
    parser.add_argument('--attr_background', type=str, default="white background", help="Description for background")
    parser.add_argument('--attr_skin', type=str, default="dark brown skin", help="Description for skin")
    parser.add_argument('--attr_accessories', type=str, default="no accessories", help="Description for accessories")
    parser.add_argument('--attr_expression', type=str, default="calm expression", help="Description for expression")
    parser.add_argument('--attr_eyebrows', type=str, default="medium eyebrows", help="Description for eyebrows")
    parser.add_argument('--attr_eyes', type=str, default="brown eyes", help="Description for eyes")
    parser.add_argument('--attr_face_shape', type=str, default="oval-shaped face", help="Description for face shape")
    parser.add_argument('--attr_hair', type=str, default="medium-length brown hair", help="Description for hair")
    parser.add_argument('--attr_mouth', type=str, default="thin lips, mouth closed, natural color lips", help="Description for mouth")

    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        os.makedirs(os.path.join(args.output_path, "generated_densepose"), exist_ok=True)

    pipe_pose, pipe_tryon, generator = prepare(args)
    print('====================== Models Loaded ===================')

    print(f"Processing inputs:\n  Cloth: {args.cloth_path}\n  Mask: {args.cloth_mask_path}")
    clo_mask_tensor = load_and_process_image(args.cloth_mask_path, args.height, args.width, is_mask=True, device=args.device)
    clo_tensor = load_and_process_image(args.cloth_path, args.height, args.width, is_mask=False, device=args.device)
    clo_tensor = clo_tensor * clo_mask_tensor
    face_tensor = load_and_process_image(args.face_path, args.height, args.width, is_mask=False, device=args.device)

    attr_list = [
        args.attr_background,
        args.attr_skin,
        args.attr_accessories,
        args.attr_expression,
        args.attr_eyebrows,
        args.attr_eyes,
        args.attr_face_shape,
        args.attr_hair,
        args.attr_mouth
    ]
    print("Attribute Prompts:", attr_list)

    # ================= Step 1: Pose Generation =================
    print("Generating Pose...")
    with torch.no_grad():
        pose_output_image = pipe_pose(
            clo=clo_tensor,
            height=args.height,
            width=args.width,
            num_inference_steps=50,
            eta=1.0
        ).images[0]

    filename = os.path.basename(args.cloth_path).split('.')[0] + "_result.jpg"
    pose_save_path = os.path.join(args.output_path, "generated_densepose", filename)
    pose_output_image.save(pose_save_path)
    generated_densepose_tensor = process_generated_densepose(
        pose_output_image, 
        device=args.device, 
        dtype=torch.float16
    )

    # ================= Step 2: Try-on Generation =================
    B, C, H, W = clo_tensor.shape
    zeros_padding = torch.zeros((B, 3, H, W), device=args.device, dtype=torch.float16)
    zeros_mask_padding = torch.zeros((B, 1, H, W), device=args.device, dtype=torch.float16)
    ones_mask_padding  = torch.ones((B, 1, H, W), device=args.device, dtype=torch.float16)
    if not args.use_face:
        clo_mask_input = torch.cat((clo_mask_tensor, zeros_mask_padding), dim=-1)
        clo_input = torch.cat((clo_tensor, zeros_padding), dim=-1)
        densepose_input = torch.cat((generated_densepose_tensor, zeros_padding), dim=-1)
    else:
        clo_mask_input = torch.cat((clo_mask_tensor, ones_mask_padding), dim=-1)
        clo_input = torch.cat((clo_tensor, face_tensor), dim=-1)
        densepose_input = torch.cat((generated_densepose_tensor, face_tensor), dim=-1)

    print("Generating Try-On...")
    image_txt = args.image_prompt + ", ".join(attr_list) + ', best quality, high quality'
    negative_prompt = 'bare, naked, nude, undressed, monochrome, lowres, bad anatomy, worst quality, low quality'
    
    with torch.no_grad():
        tryon_output = pipe_tryon(
            device=args.device,
            densepose=generated_densepose_tensor,
            clo=clo_input,
            clo_mask=clo_mask_input,
            image_txt=image_txt,
            parse_txt=attr_list,
            null_prompt='',
            negative_prompt=negative_prompt,
            width=args.width,
            height=args.height,
            guidance_scale=args.guidance_scale,
            generator=generator,
            num_inference_steps=40,
        ).images
        
        final_image = tryon_output[0].crop((0, 0, args.width, args.height))
        save_path = os.path.join(args.output_path, filename)
        final_image.save(save_path)
        print(f"Result saved to {save_path}")