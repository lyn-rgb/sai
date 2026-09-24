import torch 
import os
import json
from safetensors.torch import load_file

from modules.fusion import FusionModel
from modules.t5 import T5EncoderModel
from modules.vae2_2 import Wan2_2_VAE
from modules.mmaudio.features_utils import FeaturesUtils
    
def init_wan_vae_2_2(ckpt_dir, rank=0):
    vae_config = {}
    vae_config['device'] = rank
    vae_pth = os.path.join(ckpt_dir, "Wan2.2-TI2V-5B/Wan2.2_VAE.pth")
    vae_config['vae_pth'] = vae_pth
    vae_model = Wan2_2_VAE(**vae_config)

    return vae_model

def init_mmaudio_vae(ckpt_dir, rank=0):
    vae_config = {}
    vae_config['mode'] = '16k'
    vae_config['need_vae_encoder'] = True

    tod_vae_ckpt = os.path.join(ckpt_dir, "MMAudio/ext_weights/v1-16.pth")
    bigvgan_vocoder_ckpt = os.path.join(ckpt_dir, "MMAudio/ext_weights/best_netG.pt")

    vae_config['tod_vae_ckpt'] = tod_vae_ckpt
    vae_config['bigvgan_vocoder_ckpt'] = bigvgan_vocoder_ckpt

    vae = FeaturesUtils(**vae_config).to(rank)

    return vae

def init_fusion_score_model_ovi(rank: int = 0, meta_init=False):
    video_config = "configs/model/dit/video.json"
    audio_config = "configs/model/dit/audio.json"
    assert os.path.exists(video_config), f"{video_config} does not exist"
    assert os.path.exists(audio_config), f"{audio_config} does not exist"

    with open(video_config) as f:
        video_config = json.load(f)

    with open(audio_config) as f:
        audio_config = json.load(f)

    if meta_init:
        with torch.device("meta"):
            fusion_model = FusionModel(video_config, audio_config)
    else:
        fusion_model = FusionModel(video_config, audio_config)
    
    params_all = sum(p.numel() for p in fusion_model.parameters())
    
    if rank == 0:
        print(
            f"Score model (Fusion) all parameters:{params_all}"
        )

    return fusion_model, video_config, audio_config

def init_text_model(ckpt_dir, rank, cpu_offload=False):
    wan_dir = os.path.join(ckpt_dir, "Wan2.2-TI2V-5B")
    text_encoder_path = os.path.join(wan_dir, "models_t5_umt5-xxl-enc-bf16.pth")
    text_tokenizer_path = os.path.join(wan_dir, "google/umt5-xxl")

    text_encoder = T5EncoderModel(
        text_len=512,
        dtype=torch.bfloat16,
        device=rank,
        checkpoint_path=text_encoder_path,
        tokenizer_path=text_tokenizer_path,
        cpu_offload=cpu_offload,
        shard_fn=None)

    return text_encoder


def load_fusion_checkpoint(model, checkpoint_path, from_meta=False, strict=False):
    if checkpoint_path and os.path.exists(checkpoint_path):
        if checkpoint_path.endswith(".safetensors"): 
            df = load_file(checkpoint_path, device="cpu")
        elif checkpoint_path.endswith(".pt"):
            try:
                df = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                df = df['module'] if 'module' in df else df
            except Exception as e:
                df = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                df = df['app']['model']
        else: 
            raise RuntimeError("We only support .safetensors and .pt checkpoints")

        missing, unexpected = model.load_state_dict(df, strict=strict, assign=from_meta)
        #print(f"Missing Keys: [{missing}]")
        #print(f"Unexpected Keys: [{unexpected}]")
        
        del df
        import gc
        gc.collect()
        print(f"Successfully loaded fusion checkpoint from {checkpoint_path}")
    else:
        raise RuntimeError(f"Fusion checkpoint does not exist: {checkpoint_path}")
    
    
def load_fusion_lora(fusion, ckpt_path, from_meta=False, strict=True, strict_missing=False):
    print("=" * 45 + " Loading LoRA Weights " + "=" * 45)
    if ckpt_path and os.path.exists(ckpt_path):
        
        if ckpt_path.endswith(".safetensors"): 
            df = load_file(ckpt_path, device="cpu")
        elif ckpt_path.endswith(".pt"):
            try:
                df = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                df = df['module'] if 'module' in df else df
            except Exception as e:
                df = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                df = df['app']['model']
        else: 
            raise RuntimeError("We only support .safetensors and .pt checkpoints")
        
        state_dict = df.get("state_dict", df)
        #print(state_dict.keys())
        #print(f"=" * 90)
        #print(state_dict.keys())
        
        model = {}
        # 训练导出（train.py 的 export_trainable_state_dict 按 requires_grad 过滤）只会包含
        # `trainable_models` 里那些层。基础融合层的投影（k_fusion/v_fusion 等）在底模里，
        # `s_*_loras` 是多人的 recipe 里被移除的死层 —— 它们「不在 ckpt 里」是正常的，
        # 不该混进缺失审计里（否则每次都会报 700+ 个 key，真正的缺失反而看不出来）。
        auditable = set()
        for tower_name in ["audio_model", "video_model"]:
            tower = getattr(fusion, tower_name, None)
            if tower is None:
                continue
            # identity embedding projection (MLP over the face / speaker embedding)
            if hasattr(tower, "ip_projection"):
                print(f"[{tower_name}] Loading IP_PROJECTION")
                for sub in ["0", "2"]:
                    if not hasattr(tower.ip_projection, sub):
                        if strict:
                            raise KeyError(f"Missing module: {tower_name}.ip_projection.{sub}")
                        continue
                    layer = getattr(tower.ip_projection, sub)
                    for param_name in ["weight", "bias"]:
                        if hasattr(layer, param_name):
                            key = f"{tower_name}.ip_projection.{sub}.{param_name}"
                            model[key] = getattr(layer, param_name)
                            auditable.add(key)
                        elif strict:
                            raise KeyError(f"Missing module: {tower_name}.ip_projection.{sub}.{param_name}")

            print(f"[{tower_name}] Loading LoRAs & IP_EMBEDDING Layer")
            for i, block in enumerate(tower.blocks):
                prefix = f"{tower_name}.blocks.{i}."
                attn = block.self_attn
                for name in ["q_loras", "k_loras", "v_loras", "o_loras", "s_q_loras", "s_k_loras", "s_v_loras", "s_o_loras"]:
                    if not hasattr(attn, name):
                        continue
                    is_trained = not name.startswith("s_")          # s_*_loras：多人 recipe 里的死层
                    for sub in ["down", "up"]:
                        key = f"{prefix}self_attn.{name}.{sub}.weight"
                        if hasattr(getattr(attn, name), sub):
                            model[key] = getattr(getattr(attn, name), sub).weight
                            if is_trained:
                                auditable.add(key)
                        elif strict:
                            raise KeyError(f"Missing module: {key}")
                # ip embedding layer
                if hasattr(attn, "ip_embedding"):
                    for param_name in ["weight", "bias"]:
                        if hasattr(attn.ip_embedding, param_name):
                            key = f"{prefix}self_attn.ip_embedding.{param_name}"
                            model[key] = getattr(attn.ip_embedding, param_name)
                            auditable.add(key)
                        elif strict:
                            raise KeyError(f"Missing module: {prefix}self_attn.ip_embedding.{param_name}")
                # fusion adapters (audio<->video cross-attention, incl. the reference-pair route)
                cross_attn = getattr(block, "cross_attn", None)
                if cross_attn is None:
                    continue
                for name in ["k_fusion_lora", "v_fusion_lora"]:
                    if not hasattr(cross_attn, name):
                        continue
                    for sub in ["down", "up"]:
                        key = f"{prefix}cross_attn.{name}.{sub}.weight"
                        if hasattr(getattr(cross_attn, name), sub):
                            model[key] = getattr(getattr(cross_attn, name), sub).weight
                            auditable.add(key)
                        elif strict:
                            raise KeyError(f"Missing module: {key}")
                for name in ["k_fusion", "v_fusion", "pre_attn_norm_fusion", "norm_k_fusion"]:
                    # `k_fusion`/`v_fusion` come from the base fusion checkpoint (frozen in the
                    # multi-person recipe) so they are not audited; `pre_attn_norm_fusion` /
                    # `norm_k_fusion` are trained, so they are.
                    if not hasattr(cross_attn, name):
                        continue
                    is_trained = name in ("pre_attn_norm_fusion", "norm_k_fusion")
                    layer = getattr(cross_attn, name)
                    for param_name in ["weight", "bias"]:
                        if hasattr(layer, param_name):
                            key = f"{prefix}cross_attn.{name}.{param_name}"
                            model[key] = getattr(layer, param_name)
                            if is_trained:
                                auditable.add(key)

        loaded, skipped = 0, []
        for k, param in state_dict.items():
            if k in model:
                if model[k].shape != param.shape:
                    if strict:
                        raise ValueError(
                            f"Shape mismatch: {k} | {model[k].shape} vs {param.shape}"
                        )
                    else:
                        skipped.append(k)
                        continue
                model[k].data.copy_(param)
                loaded += 1
            else:
                if strict and "pipe.speaker_extractor" not in k:
                    raise KeyError(f"Unexpected key in ckpt: {k}")

        # Audit: a loadable adapter key that is absent from the checkpoint silently keeps its
        # initialization (zero-init for LoRA `up`, but *random* for e.g. `ip_projection`), and the
        # only symptom during training is a slightly worse loss. Report it loudly.
        # Starting a new fine-tune from an older checkpoint legitimately misses newly added layers,
        # hence `strict_missing` is opt-in.
        missing = sorted(k for k in auditable if k not in state_dict)
        not_audited = sorted(k for k in model.keys() if k not in auditable and k not in state_dict)
        if missing:
            preview = ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else "")
            msg = (f"{len(missing)} loadable key(s) not found in {os.path.basename(ckpt_path)} "
                   f"({loaded} loaded): {preview}")
            if strict_missing:
                raise KeyError(msg)
            print(f"[Warning] {msg}")
            print("[Warning] The layers above keep their current initialization. "
                  "If this is not intended, check the key names against the checkpoint.")
        print(f"Loaded {loaded} adapter tensor(s) from {os.path.basename(ckpt_path)}, "
              f"skipped {len(skipped)}, missing {len(missing)}"
              + (f"（另有 {len(not_audited)} 个属于底模/未参与训练的层，不算缺失）" if not_audited else "") + ".")
    else:
        raise RuntimeError(f"LoRA checkpoint does not exist: {ckpt_path}")
    
    print("=" * 45 + " Loading LoRA Weights " + "=" * 45)
