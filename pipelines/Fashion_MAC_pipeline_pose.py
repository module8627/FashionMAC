
from typing import List, Optional, Tuple, Union
from diffusers import ImagePipelineOutput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import *
from diffusers import DDIMPipeline
logger = logging.get_logger(__name__)  


class Fashion_MAC_pose(DDIMPipeline):
    _optional_components = []

    def __init__(
            self,
            vae_clo,
            vae_pose,
            unet,
            scheduler
        ):
        super().__init__(unet, scheduler)
        self.vae_clo = vae_clo
        self.vae_pose = vae_pose
        self.register_modules(
            vae_clo=vae_clo,
            vae_pose=vae_pose,
            unet=unet,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae_clo.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

   

    @torch.no_grad()
    def __call__(
        self,
        clo,
        height,
        width,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0,
        num_inference_steps: int = 50,
        use_clipped_model_output: Optional[bool] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers. A value of `0` corresponds to
                DDIM and `1` corresponds to DDPM.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            use_clipped_model_output (`bool`, *optional*, defaults to `None`):
                If `True` or `False`, see documentation for [`DDIMScheduler.step`]. If `None`, nothing is passed
                downstream to the scheduler (use `None` for schedulers which don't support this argument).
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Example:

        ```py
        >>> from diffusers import DDIMPipeline
        >>> import PIL.Image
        >>> import numpy as np

        >>> # load model and scheduler
        >>> pipe = DDIMPipeline.from_pretrained("fusing/ddim-lsun-bedroom")

        >>> # run pipeline in inference (sample random noise and denoise)
        >>> image = pipe(eta=0.0, num_inference_steps=50)

        >>> # process image to PIL
        >>> image_processed = image.cpu().permute(0, 2, 3, 1)
        >>> image_processed = (image_processed + 1.0) * 127.5
        >>> image_processed = image_processed.numpy().astype(np.uint8)
        >>> image_pil = PIL.Image.fromarray(image_processed[0])

        >>> # save image
        >>> image_pil.save("test.png")
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        image_shape = (
            batch_size,
            3,
            height//self.vae_scale_factor,
            width//self.vae_scale_factor,
        )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        image = randn_tensor(image_shape, generator=generator, device=self._execution_device, dtype=self.unet.dtype)
        clo = clo.to(
            dtype=self.vae_clo.dtype, device=self.vae_clo.device
        )
        with torch.no_grad():
            latent_clo = self.vae_clo.encode(clo).latent_dist.mean
            latent_clo = latent_clo * 0.18215  # (b, 4, h, w)

        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            latent_input = torch.cat([image, latent_clo], dim=1)
            model_output = self.unet(latent_input, t).sample

            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to η in paper and should be between [0, 1]
            # do x_t -> x_t-1
            image = self.scheduler.step(
                model_output, t, image, eta=eta, use_clipped_model_output=use_clipped_model_output, generator=generator
            ).prev_sample
        with torch.no_grad():
            image = self.vae_pose.post_quant_conv(image)
            image = self.vae_pose.decoder(image)
        image = image.to(dtype=torch.float32)
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
