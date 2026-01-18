from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import is_accelerate_available
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import *
from adapter.attention_processor import CAttnProcessor2_0_RADA
from torchvision.transforms import Resize, InterpolationMode
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class Fashion_MAC(StableDiffusionPipeline):
    _optional_components = []

    def __init__(
            self,
            vae,
            unet,
            tokenizer, 
            text_encoder,
            context_num,
            scheduler: Union[
                DDIMScheduler,
                PNDMScheduler,
                LMSDiscreteScheduler,
                EulerDiscreteScheduler,
                EulerAncestralDiscreteScheduler,
                DPMSolverMultistepScheduler,
            ],
            safety_checker: StableDiffusionSafetyChecker,
            feature_extractor: CLIPImageProcessor,
    ):
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler, safety_checker, feature_extractor)
        self.context_num = context_num
        self.register_modules(
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                    hasattr(module, "_hf_hook")
                    and hasattr(module._hf_hook, "execution_device")
                    and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

        # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_prompt

    def encode_prompt(
            self,
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            lora_scale: Optional[float] = None,
            clip_skip: Optional[int] = None,
    ):
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if not USE_PEFT_BACKEND:
                adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
            else:
                scale_lora_layers(self.text_encoder, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                    text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1: -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            if clip_skip is None:
                prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
                prompt_embeds = prompt_embeds[0]
            else:
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True
                )
                # Access the `hidden_states` first, that contains a tuple of
                # all the hidden states from the encoder layers. Then index into
                # the tuple to access the hidden states from the desired layer.
                prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
                # We also need to apply the final LayerNorm here to not mess with the
                # representations. The `last_hidden_states` that we typically use for
                # obtaining the final prompt representations passes through the LayerNorm
                # layer.
                prompt_embeds = self.text_encoder.text_model.final_layer_norm(prompt_embeds)

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        if isinstance(self, LoraLoaderMixin) and USE_PEFT_BACKEND:
            # Retrieve the original scale by scaling back the LoRA layers
            unscale_lora_layers(self.text_encoder, lora_scale)

        return prompt_embeds, negative_prompt_embeds

    def prepare_latents(
            self,
            batch_size,
            num_channels_latents,
            width,
            height,
            dtype,
            device,
            generator,
            latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_condition(
            self,
            cond_image,
            width,
            height,
            device,
            dtype,
            do_classififer_free_guidance=False,
    ):
        image = self.cond_image_processor.preprocess(
            cond_image, height=height, width=width
        ).to(dtype=torch.float32)

        image = image.to(device=device, dtype=dtype)

        if do_classififer_free_guidance:
            image = torch.cat([image] * 2)

        return image

    def get_image_embeds(self, clip_image=None):
        with torch.no_grad():
            # clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=torch.float16)).image_embeds
            clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=torch.float16),
                                                   output_hidden_states=True).hidden_states[-2]
            image_prompt_embeds = self.image_proj_model(clip_image_embeds)
            uncond_clip_image_embeds = self.image_encoder(
                torch.zeros_like(clip_image).to(self.device, dtype=torch.float16), output_hidden_states=True
            ).hidden_states[-2]
            uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds

    def set_scale(self, scale):
        for attn_processor in self.unet.attn_processors.values():
            if isinstance(attn_processor, CAttnProcessor2_0_RADA):
                attn_processor.scale = scale


    @torch.no_grad()
    def __call__(
            self,
            densepose,
            clo,
            clo_mask,
            image_txt,
            parse_txt,
            negative_prompt,
            width,
            height,
            num_inference_steps,
            device,
            guidance_scale=7.5,
            unet_inchannels = 4,
            num_images_per_prompt=1,
            num_samples=1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            clip_skip: Optional[int] = None,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs,
    ):
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        device = self._execution_device
        self._cross_attention_kwargs = cross_attention_kwargs
        self._clip_skip = clip_skip
        do_classifier_free_guidance = guidance_scale > 1.0

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        batch_size = 1

        text_encoder_lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            image_txt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self.clip_skip,
        )

        parse_txt_embeds, _ = self.encode_prompt(
            parse_txt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            [negative_prompt for _ in range(self.context_num)],
            prompt_embeds=None,
            negative_prompt_embeds=None,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self.clip_skip,
        )
        
        num_channels_latents = unet_inchannels
        latents_noisy = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            width * 2,
            height,
            prompt_embeds.dtype,
            device,
            generator,
        )

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        clo = clo.to(
            dtype=self.vae.dtype, device=self.vae.device
        )
        latent_clo = self.vae.encode(clo).latent_dist.mean
        latent_clo = latent_clo * 0.18215  # (b, 4, h, w)
        
        densepose = densepose.to(
            dtype=self.vae.dtype, device=self.vae.device
        )
        latent_densepose = self.vae.encode(densepose).latent_dist.mean
        latent_densepose = latent_densepose * 0.18215 
        
        latent_clo_mask = Resize(latent_clo.shape[-2:], InterpolationMode.NEAREST)(clo_mask.float()).to(dtype=self.vae.dtype, device=self.vae.device)
        latents_cond = torch.cat([latent_densepose, latent_clo, latent_clo_mask], dim=1).to(device=device, dtype=prompt_embeds.dtype)
        
        latent_h = latents_cond.shape[-2]
        latent_w = latents_cond.shape[-1]

        if do_classifier_free_guidance:
            latent_densepose_uncond = torch.cat([latent_densepose[:, :, :, :latent_w//2], torch.full(
                [1, 4, latent_h, latent_w//2], fill_value=0.0).to(dtype=self.vae.dtype, device=self.vae.device)], dim=-1)
            latent_clo_uncond = torch.cat([latent_clo[:, :, :, :latent_w//2], torch.full(
                [1, 4, latent_h, latent_w//2], fill_value=0.0).to(dtype=self.vae.dtype, device=self.vae.device)], dim=-1)
            latent_clo_mask_uncond = torch.cat([latent_clo_mask[:, :, :, :latent_w//2], torch.zeros_like(
                latent_clo_mask[:, :, :, :latent_w//2])], dim=-1).to(dtype=self.vae.dtype, device=self.vae.device)
            latents_uncond = torch.cat([latent_densepose_uncond, latent_clo_uncond, latent_clo_mask_uncond], dim=1).to(device=device, dtype=prompt_embeds.dtype)
            
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            pred_mask = None
            for i, t in enumerate(timesteps):
                latent_model_input = (
                    torch.cat([latents_noisy] * 2) if do_classifier_free_guidance else latents_noisy
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )
                
                latent_model_input_cond = torch.cat([latent_model_input[0], latents_cond[0]], dim=0)
                latent_model_input_uncond = torch.cat([latent_model_input[1], latents_uncond[0]], dim=0)

                timestep_cond = None
                if self.unet.config.time_cond_proj_dim is not None:
                    guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(
                        batch_size * num_images_per_prompt)
                    timestep_cond = self.get_guidance_scale_embedding(
                        guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
                    ).to(device=device, dtype=latents_noisy.dtype)

                noise_pred = self.unet(
                    latent_model_input_cond.unsqueeze(0),
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs={
                        "pre_step_mask": pred_mask,
                        "parse_encoder_hidden_states": parse_txt_embeds,
                    },
                    timestep_cond=timestep_cond,
                    added_cond_kwargs=None,
                    return_dict=False,
                )[0]
                for key in self.unet.attn_processors.keys():
                    if hasattr(self.unet.attn_processors[key], "now_step_pred_mask") and \
                            self.unet.attn_processors[key].now_step_pred_mask is not None:
                        pred_mask = self.unet.attn_processors[key].now_step_pred_mask

                noise_pred_uncond = self.unet(
                    latent_model_input_uncond.unsqueeze(0),
                    t,
                    encoder_hidden_states=negative_prompt_embeds,
                    timestep_cond=timestep_cond,
                    added_cond_kwargs=None,
                    return_dict=False,
                )[0]
                
                if do_classifier_free_guidance:
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred - noise_pred_uncond
                    )
                    
                latents_noisy = self.scheduler.step(
                    noise_pred, t, latents_noisy, **extra_step_kwargs, return_dict=False
                )[0]

                if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents_noisy)

        # Post-processing
        image = self.vae.decode(latents_noisy / self.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
        do_denormalize = [True] * image.shape[0]
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None)
