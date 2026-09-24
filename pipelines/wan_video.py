import torch, types, gc, time
from PIL import Image
from einops import repeat
from typing import Optional, Union
from einops import rearrange
import numpy as np
import cv2
from tqdm import tqdm
from typing import Optional
from typing_extensions import Literal
import os
from typing import List, Tuple
import PIL
from utils.model_loading_utils import (
    init_fusion_score_model_ovi, 
    init_text_model, 
    init_mmaudio_vae, 
    init_wan_vae_2_2, 
    load_fusion_checkpoint,
    load_fusion_lora
)
from modules.utils import load_state_dict
from modules.fusion import FusionModel
from modules.model import WanModel
from modules.model import WanRMSNorm, sinusoidal_embedding_1d
from modules.t5 import T5EncoderModel as WanTextEncoder
from modules.t5 import (
    T5RelativeEmbedding,
    T5LayerNorm,
)
from modules.vae2_2 import RMS_norm, CausalConv3d, Upsample
from modules.clip import CLIPModel as WanImageEncoder
from schedulers.flow_match import FlowMatchScheduler
# NOTE，这里不再 import face_cropper：训练路径不用裁剪（参考脸由
# `dataset/extract_ref_face_feats.py` 离线裁好），而它的依赖链（内联 insightface / onnxruntime）
# 一缺文件就会把整个训练 import 拖死。推理侧需要裁剪时在 `ovi_fusion_engine` 里按需导入。
# audio embedder
from modules.ns3_codec.speaker_extractor import SpeakerExtractor
# asr model
from modules.glm_asr.glmasr import GLMASR

from vram_management import (
    enable_vram_management,
    AutoWrappedModule,
    AutoWrappedLinear,
    WanAutoCastLayerNorm,
)
from distributed_comms.parallel_states import initialize_sequence_parallel_state
from lora import GeneralLoRALoader


def clean_cache():
    gc.collect()
    torch.cuda.empty_cache()


class WanVideoPipeline(torch.nn.Module):
    def __init__(self, config, resume_from=None, meta_init=True, device="cuda", torch_dtype=torch.bfloat16):
        super().__init__()
        self.device = device
        #print(f"{self.device}")
        self.torch_dtype = torch_dtype
        # The following parameters are used for shape check.
        self.height_division_factor = config.get("height_division_factor", 16)
        self.width_division_factor = config.get("width_division_factor", 16)
        self.time_division_factor = config.get("time_division_factor", 4)
        self.time_division_remainder = config.get("time_division_remainder", 1)
        self.vram_management_enabled = False

        self.config = config
        self.cpu_offload = config.get("cpu_offload", False)
        self.use_self_lora = config.get("self_lora", False)
        self.use_contrastive = config.get("use_contrastive", True)
        self.contrastive_weight = config.get("contrastive_weight", 0.1)
        self.contrastive_cutoff = config.get("contrastive_cutoff", 0.5)
        self.time_shift = config.get("time_shift", 5)

        # init env before construct models
        if config.use_sp:
            self.initialize_usp()

        # init model
        model, video_config, audio_config = init_fusion_score_model_ovi(rank=device, meta_init=meta_init)
        clean_cache()
        
        fp8 = config.get("fp8", False)
        if not meta_init:
            if not fp8:
                model = model.to(dtype=self.torch_dtype)
            model = model.to(device=device if not self.cpu_offload else "cpu")
        
        # load vaes
        vae_model_video = init_wan_vae_2_2(config.ckpt_dir, rank=device)
        vae_model_video.model.requires_grad_(False).eval()
        vae_model_video.model = vae_model_video.model.bfloat16()
        self.video_vae = vae_model_video
        clean_cache()

        vae_model_audio = init_mmaudio_vae(config.ckpt_dir, rank=device)
        vae_model_audio.requires_grad_(False).eval()
        self.audio_vae = vae_model_audio.bfloat16()
        clean_cache()

        # load text encoder
        self.text_model = init_text_model(config.ckpt_dir, rank=device, cpu_offload=self.cpu_offload)
        if config.get("shard_text_model", False):
            raise NotImplementedError("Sharding text model is not implemented yet.")
        if self.cpu_offload:
            self.offload_to_cpu(self.text_model.model)
        clean_cache()

        # load image encoder (clip)
        self.image_encoder: WanImageEncoder = None
        
        # audio embedder
        speaker_extractor = SpeakerExtractor(
            ckpt_path=config.get("audio_embedder_ckpt_dir", "../weights/naturalspeech3_facodec"), 
            device=self.device,
            dtype=self.torch_dtype
        )
        speaker_extractor.requires_grad_(False).eval()
        self.speaker_extractor = speaker_extractor
        clean_cache()
        
        # asr model to fix prompt error.
        # NOTE, the rewrite replaces everything between the first `<S>` and the last `<E>` (the
        # regex is greedy), which deletes the speaker tags of a multi-person caption, so multi-person
        # runs set `fix_prompt_with_asr: false` and feed the annotated transcripts straight through.
        self.use_asr_fix_prompt = bool(config.get("fix_prompt_with_asr", True))
        self.asr_model = None
        if self.use_asr_fix_prompt:
            self.asr_model = GLMASR(
                ckpt_path=config.get("glm_asr_ckpt_path", "./ckpts/GLM_ASR/GLM-ASR-Nano-2512"),
                device=self.device,
                max_new_tokens=config.get("max_new_tokens", 128),
            )
            clean_cache()
        
        # load fusion checkpoint
        checkpoint_path = os.path.join(config.ckpt_dir, config.ckpt_name)
        if not os.path.exists(checkpoint_path):
            raise RuntimeError(f"No fusion checkpoint found in {config.ckpt_dir}")
        load_fusion_checkpoint(model, checkpoint_path=checkpoint_path, from_meta=meta_init)

        if meta_init:
            if not fp8:
                model = model.to(dtype=self.torch_dtype)
            model = model.to(device=device if not self.cpu_offload else "cpu")
            model.set_rope_params()
        
        # init lora
        face_ip_emb_dim = config.get("face_ip_emb_dim", 512)
        audio_ip_emb_dim = config.get("audio_ip_emb_dim", 256)
        # multi-person reference conditioning: `n_refs` reference persons per sample, and the
        # reference-pair audio<->video fusion that binds each reference face to its voice
        model.n_refs = int(config.get("n_refs", 1))
        model.use_ref_av_fusion = bool(config.get("use_ref_av_fusion", False))
        model.init_lora(
            self_lora=self.use_self_lora,
            train=True,
            vid_ip_emb_dim=face_ip_emb_dim,
            audio_ip_emb_dim=audio_ip_emb_dim,
            use_ref_av_fusion=model.use_ref_av_fusion,
            fusion_lora_rank=int(config.get("fusion_lora_rank", 128)),
        )
        # load lora weights if provided
        if resume_from is not None:
            print(f"Loading lora weights from {resume_from}")
            load_fusion_lora(model, ckpt_path=resume_from, strict=False)
        
        self.model = model
        clean_cache()
        
        # init scheduler
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)

        # other params
        self._patch_size_h, self._patch_size_w = self.model.video_model.patch_size[1], self.model.video_model.patch_size[2]

        # forward function
        self.model_fn = model_fn
        
    def switch_to_train(self):
        # set trainable params
        self.model.train()
        self.scheduler.set_timesteps(1000, training=True)
        self.freeze_except(self.model, [] if self.config.get("trainable_models", None) is None else self.config.get("trainable_models", None).split(","))

    def offload_to_cpu(self, model):
        model = model.cpu()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        return model
    
    def get_vram(self):
        return torch.cuda.mem_get_info(self.device)[1] / (1024**3)

    def freeze_except(self, model, model_names):
        for name, sub_model in model.named_children():
            if name in model_names:
                sub_model.train()
                sub_model.requires_grad_(True)
            else:
                sub_model.eval()
                sub_model.requires_grad_(False)
                self.freeze_except(sub_model, model_names)
    
    @torch.no_grad()
    def crop_image(self, images: torch.Tensor):
        # NOTE，训练路径不会走到这里（`self.cropper` / `self.face_embedder` 只在推理侧初始化）
        assert getattr(self, "cropper", None) is not None, \
            "crop_image 需要先初始化 self.cropper（推理侧才有），训练路径不应调用"
        # image, tensor, [-1, 1], Bx3xHxW or Bx3x1xHxW,
        assert (images.dim() == 4) or (images.dim() == 5 and images.shape[2] == 1)
        has_t_dim = (images.dim() == 5)
        if has_t_dim:
            images = images.squeeze(2)
        _images = images.permute(0, 2, 3, 1)  # BxHxWx3
        _images = ((_images + 1) / 2) * 255    # [0, 255]
        _images = _images.cpu().numpy()
        cropped_images = []
        face_embs = []
        try:
            for image in _images:
                cropped_image = self.cropper.crop_source_image(image, self.cropper.crop_cfg)["img_crop"]
                #print(f"cropped_image shape: {cropped_image.shape}, value range: [{cropped_image.min()}, {cropped_image.max()}]")
                # get face embeddings, RGB -> BGR
                face_info = self.face_embedder.get(cv2.cvtColor(cropped_image.copy(), cv2.COLOR_RGB2BGR))
                face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1] # only use the maximum face
                face_emb = face_info['embedding']
                face_embs.append(torch.from_numpy(face_emb))
                
                # get cropped face
                cropped_image = torch.from_numpy(cropped_image).float()
                cropped_image = (cropped_image.permute(2, 0, 1) / 255) * 2 - 1
                cropped_images.append(cropped_image)
            face_embs = torch.stack(face_embs)  # Bx512
            cropped_images = torch.stack(cropped_images)
        except Exception as e:
            print(f"Error <{e}> occurrs during crop images.")
            cropped_images = images
            face_embs = None
        if has_t_dim:
            cropped_images = cropped_images.unsqueeze(2)
        return cropped_images, face_embs    # [-1, 1], Bx3xHxW or Bx3x1xHxW
                
    @torch.no_grad()
    def preprocess_inputs(self, inputs: dict) -> dict:
        outputs = {}
        
        # 1. process input video
        #print(f"inputs: {inputs.keys()}")
        video = inputs["video"].to(dtype=torch.bfloat16, device=self.device)    # B, 3, T, H, W, [-1, 1]
        #print(f"video shape: {video.shape} [{video.min()}, {video.max()}]")
        outputs["batch_size"] = video.shape[0]
        video_input_latents = self.video_vae.wrapped_encode(video).to(dtype=self.torch_dtype, device=self.device)
        outputs["video_input_latents"] = video_input_latents    # B, C, T', H', W'
        
        # 2. process input audio
        audio = inputs["audio"].to(dtype=torch.float32, device=self.device)    # B, L [-1, 1], mel convertor does not support bfloat16
        #print(f"audio shape: {audio.shape} [{audio.min()}, {audio.max()}]")
        audio_input_latents = self.audio_vae.wrapped_encode(audio).to(dtype=self.torch_dtype, device=self.device).transpose(-1, -2)
        outputs["audio_input_latents"] = audio_input_latents # B, L', D

        # 3. sample noises
        video_noise = torch.randn_like(video_input_latents).to(dtype=self.torch_dtype, device=self.device)
        audio_noise = torch.randn_like(audio_input_latents).to(dtype=self.torch_dtype, device=self.device)
        #print(f"video noise shape: {video_noise.shape} [{video_noise.min()}, {video_noise.max()}], audio noise shape: {audio_noise.shape} [{audio_noise.min()}, {audio_noise.max()}]")
        outputs["video_noise"] = video_noise
        outputs["audio_noise"] = audio_noise

        # 4. process prompts
        if self.use_asr_fix_prompt:
            prompts = self.asr_model.fix_prompt(inputs["prompts"], inputs["ori_audio"])
        else:
            prompts = inputs["prompts"]
        #print(f"[Ori Prompt] {inputs['prompts'][0]} \n" + f"[Fix Prompt] {prompts[0]} \n")
        
        context = self.text_model(prompts, device=self.device)
        outputs["audio_context"] = context
        outputs["video_context"] = context
        
        outputs["video_seq_len"] = video_noise.shape[2] * video_noise.shape[3] * video_noise.shape[4] // (self._patch_size_h * self._patch_size_w)
        outputs["audio_seq_len"] = audio_noise.shape[1]
        outputs["first_frame_is_clean"] = False   # not i2v
        
        # 5. process ip image, B, 3, N, H, W (N reference faces, one per person)
        if "ip_image" in inputs:
            ip_image = inputs["ip_image"].to(dtype=torch.bfloat16, device=self.device)
            if ip_image.dim() == 4:                     # legacy single reference
                ip_image = ip_image.unsqueeze(2)
            b, _, n_refs, h, w = ip_image.shape
            # NOTE, encode every reference on its own: the video VAE only keeps the first frame of
            # each window (`vae2_2.py`, `iter_ = 1 + (t - 1) // 4`), so encoding several reference
            # frames in one call would silently drop all but the first one.
            flat = ip_image.permute(0, 2, 1, 3, 4).reshape(b * n_refs, 3, 1, h, w)
            latents = self.video_vae.wrapped_encode(flat).to(dtype=self.torch_dtype, device=self.device)
            c_lat, h_lat, w_lat = latents.shape[1], latents.shape[3], latents.shape[4]
            # B, C, N, H', W' - reference tokens are read in person-major order downstream
            latents = latents.reshape(b, n_refs, c_lat, 1, h_lat, w_lat).permute(0, 2, 3, 1, 4, 5)
            outputs["ip_image_latents"] = latents.reshape(b, c_lat, n_refs, h_lat, w_lat)
            # one identity vector per person, reduced to the legacy single-vector path so the
            # per-person binding is carried by the reference tokens and the reference-pair fusion
            ip_image_embs = inputs["ip_image_embs"].to(dtype=torch.bfloat16, device=self.device)
            outputs["ip_image_embs"] = ip_image_embs.mean(dim=-2) if ip_image_embs.dim() == 3 else ip_image_embs
        else:
            print(f"[Warning] ip_image do not exists.")

        # 6. process ip audio, B, N, L (N reference voices, one per person)
        if "ip_audio" in inputs:
            ip_audio = inputs["ip_audio"].to(dtype=torch.float32, device=self.device)
            if ip_audio.dim() == 2:                     # legacy single reference
                ip_audio = ip_audio.unsqueeze(1)
            b, n_refs, length = ip_audio.shape
            # likewise encode each voice on its own: the audio VAE bottleneck is a global attention
            # block, so concatenating two voices into one waveform would entangle their timbres
            flat = ip_audio.reshape(b * n_refs, length)
            latents = self.audio_vae.wrapped_encode(flat).to(dtype=self.torch_dtype, device=self.device).transpose(-1, -2)
            outputs["ip_audio_latents"] = latents.reshape(b, n_refs * latents.size(1), latents.size(2))
        else:
            print(f"[Warning] ip_audio do not exists.")

        # 7. process ip audio spk_embs
        if "ori_ip_audio" in inputs:
            # ori_ip_audio, is the waveform without normalization
            try:
                ori_ip_audio = inputs["ori_ip_audio"].to(dtype=self.torch_dtype, device=self.device)
                if ori_ip_audio.dim() == 1:
                    ori_ip_audio = ori_ip_audio.unsqueeze(0)
                if ori_ip_audio.dim() == 2:
                    b, n_refs = ori_ip_audio.size(0), 1
                    ori_ip_audio = ori_ip_audio.unsqueeze(1)
                else:
                    b, n_refs = ori_ip_audio.size(0), ori_ip_audio.size(1)
                    ori_ip_audio = ori_ip_audio.reshape(b * n_refs, 1, ori_ip_audio.size(2))
                spk_embs = self.speaker_extractor(ori_ip_audio).to(dtype=torch.bfloat16, device=self.device)
                # one timbre vector per person, reduced to the legacy single-vector path
                outputs["ip_audio_embs"] = spk_embs.reshape(b, n_refs, -1).mean(dim=1)
            except Exception as e:
                print(f"[Error] `{e}`s occured during extracting audio embeddings.")
        else:
            print(f"[Warning] speaker embeddings do not exists.")
            
        return outputs
    
    @torch.no_grad()
    def reconstruct_audio(self, inputs):
        # audio_tensor, [-1, 1], shape: B,L
        audio = inputs["audio"].to(dtype=torch.float32, device=self.device)    # B, L [-1, 1], mel convertor does not support bfloat16
        audio_latents = self.audio_vae.wrapped_encode(audio).to(dtype=self.torch_dtype, device=self.device).transpose(-1, -2)
        audio_latents = audio_latents.transpose(1, 2)  # 1, c, l
        rec_audio = self.audio_vae.wrapped_decode(audio_latents)
        rec_audio = rec_audio.squeeze().cpu().float().numpy()
        
        # ip audio
        ip_audio = inputs["ip_audio"].to(dtype=torch.float32, device=self.device)    # B, L    
        ip_audio_latents = self.audio_vae.wrapped_encode(ip_audio).to(dtype=self.torch_dtype, device=self.device).transpose(-1, -2)
        ip_audio_latents = ip_audio_latents.transpose(1, 2)  # 1, c, l
        rec_ip_audio = self.audio_vae.wrapped_decode(ip_audio_latents)
        rec_ip_audio = rec_ip_audio.squeeze().cpu().float().numpy()
        
        return rec_audio, rec_ip_audio
        
    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(
            *args, **kwargs
        )
        if device is not None:
            self.device = device
        if dtype is not None:
            self.torch_dtype = dtype
        super().to(*args, **kwargs)
        return self

    def load_lora(self, module, path, alpha=1):
        loader = GeneralLoRALoader(torch_dtype=self.torch_dtype, device=self.device)
        lora = load_state_dict(path, torch_dtype=self.torch_dtype, device=self.device)
        loader.load(module, lora, alpha=alpha)

    def training_loss(self, **inputs):
        batch_size = inputs["batch_size"]
        max_timestep_boundary = int(
            inputs.get("max_timestep_boundary", 1) * self.scheduler.num_train_timesteps
        )
        min_timestep_boundary = int(
            inputs.get("min_timestep_boundary", 0) * self.scheduler.num_train_timesteps
        )
        timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (batch_size,))
        timestep = self.scheduler.timesteps[timestep_id].to(
            dtype=self.torch_dtype, device=self.device
        )

        inputs["video_latents"] = self.scheduler.add_noise(
            inputs["video_input_latents"], inputs["video_noise"], timestep
        )
        training_video_target = self.scheduler.training_target(
            inputs["video_input_latents"], inputs["video_noise"], timestep
        )
        inputs["audio_latents"] = self.scheduler.add_noise(
            inputs["audio_input_latents"], inputs["audio_noise"], timestep
        )
        training_audio_target = self.scheduler.training_target(
            inputs["audio_input_latents"], inputs["audio_noise"], timestep
        )

        noise_pred_vid, noise_pred_aud = self.model_fn(**inputs, timestep=timestep)
        
        loss_video = torch.nn.functional.mse_loss(noise_pred_vid.float(), training_video_target.float())
        loss_audio = torch.nn.functional.mse_loss(noise_pred_aud.float(), training_audio_target.float())
        loss_video = loss_video * self.scheduler.training_weight(timestep)
        loss_audio = loss_audio * self.scheduler.training_weight(timestep)
        
        loss = (loss_video + loss_audio).mean()
        losses = {"loss_video": loss_video, "loss_audio": loss_audio}
        
        # contrastive loss
        if self.use_contrastive:
            inputs["ip_image_latents"] = None
            inputs["ip_audio_latents"] = None
            with torch.no_grad():
                uncond_noise_pred_vid, uncond_noise_pred_aud = self.model_fn(**inputs, timestep=timestep)
                
            loss_video_contrastive = torch.nn.functional.mse_loss(noise_pred_vid.float(), uncond_noise_pred_vid.detach().float())
            loss_audio_contrastive = torch.nn.functional.mse_loss(noise_pred_aud.float(), uncond_noise_pred_aud.detach().float())
            sigma = timestep / self.scheduler.num_train_timesteps   # [0, 1]
            loss_weight = sigma / (self.time_shift - (self.time_shift - 1) * sigma + 1e-5)
            loss_weight[sigma < self.contrastive_cutoff] = 0.0
            loss_weight = loss_weight * self.contrastive_weight * self.scheduler.training_weight(timestep)
                
            loss = loss - (loss_weight * (loss_video_contrastive + loss_audio_contrastive)).mean()
            
            losses["loss_video_contrastive"] = loss_video_contrastive
            losses["loss_audio_contrastive"] = loss_audio_contrastive
            
        losses["loss"] = loss        
        
        return losses

    def enable_vram_management(
        self, num_persistent_param_in_dit=None, vram_limit=None, vram_buffer=0.5
    ):
        self.vram_management_enabled = True
        if num_persistent_param_in_dit is not None:
            vram_limit = None
        else:
            if vram_limit is None:
                vram_limit = self.get_vram()
            vram_limit = vram_limit - vram_buffer
        if self.text_model is not None:
            dtype = next(iter(self.text_model.parameters())).dtype
            enable_vram_management(
                self.text_model,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Embedding: AutoWrappedModule,
                    T5RelativeEmbedding: AutoWrappedModule,
                    T5LayerNorm: AutoWrappedModule,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.model is not None:
            dtype = next(iter(self.model.parameters())).dtype
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.model,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: WanAutoCastLayerNorm,
                    WanRMSNorm: AutoWrappedModule,
                    torch.nn.Conv2d: AutoWrappedModule,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                max_num_param=num_persistent_param_in_dit,
                overflow_module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.audio_vae is not None:
            dtype = next(iter(self.vae.parameters())).dtype
            enable_vram_management(
                self.vae,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    RMS_norm: AutoWrappedModule,
                    CausalConv3d: AutoWrappedModule,
                    Upsample: AutoWrappedModule,
                    torch.nn.SiLU: AutoWrappedModule,
                    torch.nn.Dropout: AutoWrappedModule,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=self.device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
            )
        if self.video_vae is not None:
            dtype = next(iter(self.vae.parameters())).dtype
            enable_vram_management(
                self.vae,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    RMS_norm: AutoWrappedModule,
                    CausalConv3d: AutoWrappedModule,
                    Upsample: AutoWrappedModule,
                    torch.nn.SiLU: AutoWrappedModule,
                    torch.nn.Dropout: AutoWrappedModule,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=self.device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
            )
        
    def initialize_usp(self):
        initialize_sequence_parallel_state(self.config.get("sp_size", 1))


def model_fn(
    model: FusionModel,
    video_latents: torch.Tensor = None,
    audio_latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    video_context: torch.Tensor = None,
    audio_context: torch.Tensor = None,
    video_seq_len: int = None,
    audio_seq_len: int = None,
    ip_image_latents: Optional[torch.Tensor] = None,
    ip_audio_latents: Optional[torch.Tensor] = None,
    ip_image_embs: Optional[torch.Tensor] = None,
    ip_audio_embs: Optional[torch.Tensor] = None,
    **kwargs,
):
    return model(
        vid=video_latents,
        audio=audio_latents,
        t=timestep,
        vid_context=video_context,
        audio_context=audio_context,
        vid_seq_len=video_seq_len,
        audio_seq_len=audio_seq_len,
        clip_fea=None,
        clip_fea_audio=None,
        y=None,
        first_frame_is_clean=False,
        slg_layer=False,
        vid_ip=ip_image_latents,
        audio_ip=ip_audio_latents,
        vid_ip_emb=ip_image_embs,
        audio_ip_emb=ip_audio_embs
    )
    
