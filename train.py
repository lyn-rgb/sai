from functools import partial
import os, torch, warnings, torchvision, argparse, json, gc, math
from PIL import Image
import random
import numpy as np
import logging
import pandas as pd
import time
from datetime import datetime
from tqdm import tqdm
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

# asr model
#from modules.glm_asr.glmasr import GLMASR
from pipelines.wan_video import WanVideoPipeline
from dataset.text_video_audio_dataset import TextAudioVideoDataset, TextAudioVideoFaceDataset
from utils.io_utils import save_video


import tempfile
from scipy.io import wavfile

def save_audio(save_path, audio_numpy, sr=16000):
    wavfile.write(save_path, sr, (audio_numpy * 32767).astype(np.int16))


def get_timestr():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    

def create_logger(logging_dir=None, is_main_process=True):
    if is_main_process:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        if logging_dir is not None:
            file_handler = logging.FileHandler(f"{logging_dir}/train.log")
            file_handler.setLevel(logging.INFO)
            file_formatter = logging.Formatter(fmt="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        else:
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(logging.INFO)
            stream_formatter = logging.Formatter(fmt="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
            stream_handler.setFormatter(stream_formatter)
            logger.addHandler(stream_handler)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler)
    return logger


def get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    else:
        return 0
    
    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    
def format_numel_str(numel: int) -> str:
    B = 1024 ** 3
    M = 1024 ** 2
    K = 1024
    if numel > B:
        return f"{numel / B:.2f} B"
    elif numel > M:
        return f"{numel / M:.2f} M"
    elif numel > K:
        return f"{numel / K:.2f} K"
    else:
        return numel


class DiffusionTrainingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_train_params = 0
        
    def to(self, *args, **kwargs):
        for name, model in self.named_children():
            model.to(*args, **kwargs)
        return self
    
    def get_num_train_params(self):
        if self.num_train_params == 0:
            for p in self.trainable_modules():
                self.num_train_params += p.numel()
        return self.num_train_params
        
    def trainable_modules(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.parameters())
        return trainable_modules
    
    def trainable_param_names(self):
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        return trainable_param_names
    
    def mapping_lora_state_dict(self, state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if "lora_A.weight" in key or "lora_B.weight" in key:
                new_key = key.replace("lora_A.weight", "lora_A.default.weight").replace("lora_B.weight", "lora_B.default.weight")
                new_state_dict[new_key] = value
            elif "lora_A.default.weight" in key or "lora_B.default.weight" in key:
                new_state_dict[key] = value
        return new_state_dict

    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        trainable_param_names = self.trainable_param_names()
        state_dict = {name: param for name, param in state_dict.items() if name in trainable_param_names}
        if remove_prefix is not None:
            state_dict_ = {}
            for name, param in state_dict.items():
                if name.startswith(remove_prefix):
                    name = name[len(remove_prefix):]
                state_dict_[name] = param
            state_dict = state_dict_
        return state_dict
    
    def transfer_data_to_device(self, data, device, torch_float_dtype=None):
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device)
                if torch_float_dtype is not None and data[key].dtype in [torch.float, torch.float16, torch.bfloat16]:
                    data[key] = data[key].to(torch_float_dtype)
        return data
    

class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        config,
        resume_from=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        seed = 42,
    ):
        super().__init__()
        set_seed(seed)
        # all ranks share same seed for model initialization
        # Load models
        self.pipe = WanVideoPipeline(config, resume_from=resume_from, meta_init=True, device=torch.cuda.current_device(), torch_dtype=torch.bfloat16)        
        self.pipe.model.set_gradient_checkpointing(True)
        
        # Training mode
        self.pipe.switch_to_train()
        #print(f"number of trainable paramters: {format_numel_str(self.get_num_train_params())}")
        
        # Store other configs
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
    
    def forward(self, inputs):
        inputs = self.pipe.preprocess_inputs(inputs)
        models = {"model": self.pipe.model}
        losses = self.pipe.training_loss(**models, **inputs)
        return losses
    

class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x):
        self.output_path = output_path + "/ckpt"
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0

    def on_step_end(self, accelerator, model, save_steps=None):
        self.num_steps += 1
        if save_steps is not None and self.num_steps % save_steps == 0:            
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")            

    def export_state_dict(self, accelerator, model):
        """The trainable weights to store, with a loud failure if there is nothing to store.

        `export_trainable_state_dict` filters by `requires_grad`, so a config whose `trainable_models`
        names do not match any module (or a `freeze_except` that froze everything) produces an **empty**
        dict - and saving it writes a ~0-byte checkpoint that only fails much later, at load time.
        """
        state_dict = accelerator.get_state_dict(model)
        unwrapped = accelerator.unwrap_model(model)
        state_dict = unwrapped.export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
        state_dict = self.state_dict_converter(state_dict)
        if not state_dict:
            trainable = len(unwrapped.trainable_param_names())
            raise RuntimeError(
                f"没有任何可训练权重可存档（requires_grad=True 的参数个数 = {trainable}）。"
                f"检查训练配置里的 trainable_models 名字是否与模块或 `WanModel.init_lora` 建出来的"
                f"子模块同名（freeze_except 只按名字匹配，对不上的会被静默冻结）")
        return state_dict

    def on_epoch_end(self, accelerator, model, epoch_id):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = self.export_state_dict(accelerator, model)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)
        gc.collect()
        torch.cuda.empty_cache()

    def on_training_end(self, accelerator, model, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")

    def save_model(self, accelerator, model, file_name):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = self.export_state_dict(accelerator, model)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)
        gc.collect()
        torch.cuda.empty_cache()


def collate_fn(data):
    batch_data = {}
    for key in data[0].keys():
        if isinstance(data[0][key], str):
            batch_data[key] = [x[key] for x in data]
        elif isinstance(data[0][key], torch.Tensor):
            batch_data[key] = torch.stack([x[key] for x in data])
        else:
            raise TypeError(f"Only suport `str` or `torch.Tensor` data.")
    return batch_data
    
    
def worker_init_fn_seed(worker_id, num_workers, rank, seed):
    worker_seed = num_workers * rank + worker_id + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    
    
def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 8,
    save_steps: int = 2000,
    num_epochs: int = 1,
    gradient_accumulation_steps: int = 1,
    debug_path = "./logs/debug",
    log_steps: int = 20,
    args = None,
):
    # set specific seed for each rank
    set_seed(42 + get_rank())
    
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        gradient_accumulation_steps = args.gradient_accumulation_steps
        debug_path = os.path.join(args.output_path, "debug")
        log_steps = args.log_steps
        
    if accelerator.is_main_process:
        if not os.path.exists(debug_path):
            os.makedirs(debug_path)
    
    logger = create_logger(logging_dir=args.output_path, is_main_process=accelerator.is_main_process)
    logger.info(f"Number of trainable paramters: {format_numel_str(model.get_num_train_params())}")
    
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=collate_fn, num_workers=num_workers, worker_init_fn=partial(worker_init_fn_seed, num_workers=num_workers, rank=get_rank(), seed=42))
    
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    
    num_samples = len(dataloader)
    loss_info = {}
    avg_losses = {}
    total_iter = 0
    for epoch_id in range(num_epochs):
        epoch_loss = 0.0
        with tqdm(total=len(dataloader), desc=f"[Epoch {epoch_id+1}/{num_epochs}]", disable=True) as pbr:            
            for data in dataloader:
                try:                                            
                    with accelerator.accumulate(model):
                        optimizer.zero_grad()
                        losses = model(data)
                        accelerator.backward(losses["loss"])
                        optimizer.step()
                        model_logger.on_step_end(accelerator, model, save_steps)
                        scheduler.step()
                        
                        loss_info["lr"] = optimizer.param_groups[0]["lr"]
                        for key, loss in losses.items():
                            loss = accelerator.reduce(loss, reduction="mean", scale=gradient_accumulation_steps)
                            loss_info[key] = loss.item()                        
                except Exception as e:
                    timestr = get_timestr()
                    with open(f"{debug_path}/errors.log", "a") as f:                        
                        f.write(f"{timestr} Rank-{dist.get_rank()}: Error <{e}>, video shape: {data['video'].shape}, audio shape: {data['audio'].shape}, ip image shape: {data['ip_image'].shape}, ip audio shape: {data['ip_audio'].shape}\n")
                    raise RuntimeError(f"{timestr} Rank-{dist.get_rank()}: Error <{e}>, video shape: {data['video'].shape}, audio shape: {data['audio'].shape}, ip image shape: {data['ip_image'].shape}, ip audio shape: {data['ip_audio'].shape}")
                
                # debug sample
                if total_iter == 0 and accelerator.is_main_process:
                    # save video & ip image
                    video_sample_path = os.path.join(debug_path, "sample_video.mp4")
                    save_video(video_sample_path, data["video"][0].cpu().numpy())
                    
                    # reference faces, [B, 3, N, H, W] with N references per sample
                    image_sample_paths = []
                    for ref_idx in range(data["ip_image"].shape[2]):
                        image_sample_path = os.path.join(debug_path, f"ip_image_{ref_idx}.jpg")
                        ip_image = data["ip_image"][0, :, ref_idx].permute(1, 2, 0)   # H,W,C
                        ip_image = ((ip_image + 1) / 2) * 255
                        ip_image = ip_image.cpu().numpy().astype(np.uint8)
                        ip_image = Image.fromarray(ip_image)
                        ip_image.save(image_sample_path)
                        image_sample_paths.append(image_sample_path)

                    # test audio
                    audio_sample_path = os.path.join(debug_path, "sample_audio.wav")
                    save_audio(audio_sample_path, data["audio"][0].cpu().numpy())
                    ip_audio_sample_paths = []
                    for ref_idx in range(data["ip_audio"].shape[1]):
                        ip_audio_sample_path = os.path.join(debug_path, f"ip_audio_{ref_idx}.wav")
                        save_audio(ip_audio_sample_path, data["ip_audio"][0, ref_idx].cpu().numpy())
                        ip_audio_sample_paths.append(ip_audio_sample_path)

                    logger.info("=" * 45 + " DEBUG INFO " + "=" * 45 + "\n"
                        + f"sample video {video_sample_path} \n"
                        + f"ip images {image_sample_paths} \n"
                        + f"sample audio {audio_sample_path} \n"
                        + f"ip audios {ip_audio_sample_paths} \n"
                        + f"video shape: {data['video'].shape} \n"
                        + f"audio shape: {data['audio'].shape} \n"
                        + f"ip image shape: {data['ip_image'].shape} \n"
                        + f"ip audio shape: {data['ip_audio'].shape} \n"
                        + f"reference slots filled (0 = padded duplicate): {data['ref_valid'].tolist()} \n"
                        + "=" * 45 + " DEBUG INFO " + "=" * 45
                    )
                    
                # logging some info
                accelerator.log(loss_info, step=total_iter)
                
                epoch_loss += loss_info['loss'] / num_samples
                for loss_name, loss_value in loss_info.items():
                    avg_losses[loss_name] = avg_losses.get(loss_name, 0.0) + loss_value
                    
                if total_iter % log_steps == 0:
                    log_str = f"{get_timestr()} [epoch: {epoch_id+1:02d}/{num_epochs:02d}] [step: {total_iter:08d}/{num_samples}] "
                    for loss_name, loss_value in avg_losses.items():
                        if total_iter == 0:
                            log_str += f"[{loss_name}: {(loss_value):06f}] "
                        else:
                            log_str += f"[{loss_name}: {(loss_value/log_steps):06f}] "
                        
                    logger.info(log_str)
                    avg_losses = {}
                    
                pbr.set_postfix(loss_info)
                total_iter += 1
                pbr.update()
        if accelerator.is_main_process:
            logger.info(f"{get_timestr()} [Epoch {epoch_id:03d}] [Average Loss: {epoch_loss}]")
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)


def launch_data_process_task(
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    accelerator = Accelerator()
    model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in tqdm(enumerate(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data, return_inputs=True)
                torch.save(data, save_path)


def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--data_root", type=str, default="", required=True, help="Base path of the dataset.")
    parser.add_argument("--meta_dir", type=str, default=None, required=True, help="Path to the metadata file of the dataset.")
    parser.add_argument("--model_config_path", type=str, default=None, required=True, help="Path to the metadata file of the dataset.")
    parser.add_argument("--resume_from_ckpt", type=str, default=None, help="Resume from ckpt")
    parser.add_argument("--height", type=int, default=480, help="Height of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=864, help="Width of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--num_frames", type=int, default=121, help="Number of frames per video. Frames are sampled from the video prefix.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.model.", help="Remove prefix in ckpt.")
    
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--find_unused_parameters", default=False, action="store_true", help="Whether to find unused parameters in DDP.")
    parser.add_argument("--log_steps", type=int, default=20, help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.")
    parser.add_argument("--save_steps", type=int, default=1000, help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.")
    parser.add_argument("--dataset_num_workers", type=int, default=8, help="Number of workers for data loading.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    return parser


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()

    config = OmegaConf.load(args.model_config_path)
    tb_log_dir = args.output_path + "/tensorboard"
    project_config = ProjectConfiguration(project_dir="./", logging_dir=tb_log_dir)
    
    accelerator = Accelerator(
        log_with="tensorboard", project_config=project_config,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    accelerator.init_trackers("sai")
    
    target_fps = 24
    # reference audio length, 96 frames = 4s (single-person recipe) and must match the inference
    # side (`ref_audio_samples`); shorter clips need a shorter reference, e.g. 48 frames = 2s
    ref_audio_frames = int(config.get("ref_audio_frames", 96))
    min_video_len = math.ceil((ref_audio_frames + target_fps/2 + args.num_frames) / target_fps)
    dataset = TextAudioVideoFaceDataset(
        data_root        = args.data_root,
        meta_dir         = args.meta_dir,
        audio_sr         = 16000,
        ref_audio_frames = ref_audio_frames,
        normalize_audio  = True,
        height           = args.height,
        width            = args.width,
        height_div       = 32,
        width_div        = 32,
        num_frames       = args.num_frames,
        min_video_len    = min_video_len,
        n_refs           = int(config.get("n_refs", 1)),
    )
    print(f"[Dataset] {len(dataset)} samples, n_refs={dataset.n_refs}, "
          f"reference audio length={dataset.ref_audio_length}")
    model = WanTrainingModule(
        config=config,
        resume_from=args.resume_from_ckpt,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt
    )
    
    # asr_model = GLMASR(
    #     ckpt_path=config.get("glm_asr_ckpt_path", "./ckpts/GLM_ASR/GLM-ASR-Nano-2512"),
    #     device=torch.cuda.current_device(),
    #     max_new_tokens=config.get("max_new_tokens", 128),
    # )
    
    launch_training_task(
        accelerator=accelerator, 
        dataset=dataset, 
        model=model, 
        model_logger=model_logger, 
        args=args
    )
    
    accelerator.end_training()