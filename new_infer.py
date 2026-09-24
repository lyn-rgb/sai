import os
import sys
import logging
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from utils.io_utils import save_video
from utils.processing_utils import format_prompt_for_filename, validate_and_process_user_prompt
from utils.utils import get_arguments
from distributed_comms.util import get_world_size, get_local_rank, get_global_rank
from distributed_comms.parallel_states import initialize_sequence_parallel_state, get_sequence_parallel_state, nccl_info
from ovi_fusion_engine import OviFusionEngine


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def get_next_sequence_number(output_dir, condition_dir):
    """获取下一个序列号"""
    condition_output_dir = os.path.join(output_dir, condition_dir)
    
    # 如果目录不存在，从1开始
    if not os.path.exists(condition_output_dir):
        return 1
    
    # 查找目录中已有的文件，提取最大序列号
    max_sequence = 0
    for filename in os.listdir(condition_output_dir):
        if filename.endswith('.mp4') or filename.endswith('.png'):
            # 尝试从文件名中提取序列号
            parts = filename.split('_')
            if parts and parts[0].isdigit():
                sequence = int(parts[0])
                if sequence > max_sequence:
                    max_sequence = sequence
    
    return max_sequence + 1


def get_config_bool(config, key, default=False):
    value = config.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def find_existing_output(output_dir, output_filename):
    output_path = os.path.join(output_dir, output_filename)
    if os.path.isfile(output_path):
        return output_path

    # Also catch outputs from earlier append-style runs where only the sequence number differs.
    filename_parts = output_filename.split("_", 1)
    if len(filename_parts) != 2 or not os.path.isdir(output_dir):
        return None
    output_suffix = filename_parts[1]
    for filename in os.listdir(output_dir):
        if not filename.endswith(".mp4"):
            continue
        existing_parts = filename.split("_", 1)
        if len(existing_parts) == 2 and existing_parts[1] == output_suffix:
            return os.path.join(output_dir, filename)
    return None


def main(config, args): 

    world_size = get_world_size()
    global_rank = get_global_rank()
    local_rank = get_local_rank()
    device = local_rank
    torch.cuda.set_device(local_rank)
    sp_size = config.get("sp_size", 1)
    assert sp_size <= world_size and world_size % sp_size == 0, "sp_size must be less than or equal to world_size and world_size must be divisible by sp_size."

    _init_logging(global_rank)

    if world_size > 1:
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=global_rank,
            world_size=world_size)
    else:
        assert sp_size == 1, f"When world_size is 1, sp_size must also be 1, but got {sp_size}."
        ## TODO: assert not sharding t5 etc...


    initialize_sequence_parallel_state(sp_size)
    logging.info(f"Using SP: {get_sequence_parallel_state()}, SP_SIZE: {sp_size}")
    
    args.local_rank = local_rank
    args.device = device
    target_dtype = torch.bfloat16

    # validate inputs before loading model to not waste time if input is not valid
    text_prompt = config.get("text_prompt")
    image_path = config.get("image_path", None)
    ip_image_path = config.get("ip_image_path", None)
    ip_audio_path = config.get("ip_audio_path", None)
    self_lora = config.get("self_lora", True)
    assert config.get("mode") in ["id2v", "t2v", "i2v", "t2i2v"], f"Invalid mode {config.get('mode')}, must be one of ['id2v', 't2v', 'i2v', 't2i2v']"
    text_prompts, image_paths, ip_image_paths, ip_audio_paths = validate_and_process_user_prompt(text_prompt, image_path, ip_image_path, ip_audio_path, mode=config.get("mode"))
    if config.get("mode") != "i2v":
        logging.info(f"mode: {config.get('mode')}, setting all image_paths, ip_image_paths and ip_audio_paths to None")
        image_paths = [None] * len(text_prompts)
    else:
        assert all(p is not None and os.path.isfile(p) for p in image_paths), f"In i2v mode, all image paths must be provided.{image_paths}"

    logging.info("Loading OVI Fusion Engine...")
    ovi_engine = OviFusionEngine(config=config, device=device, target_dtype=target_dtype, self_lora=self_lora)
    logging.info("OVI Fusion Engine loaded!")
    
    output_dir = config.get("output_dir", "./outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Load CSV data
    all_eval_data = list(zip(text_prompts, image_paths, ip_image_paths, ip_audio_paths))

    # Get SP configuration
    use_sp = get_sequence_parallel_state()
    if use_sp:
        sp_size = nccl_info.sp_size
        sp_rank = nccl_info.rank_within_group
        sp_group_id = global_rank // sp_size
        num_sp_groups = world_size // sp_size
    else:
        # No SP: treat each GPU as its own group
        sp_size = 1
        sp_rank = 0
        sp_group_id = global_rank
        num_sp_groups = world_size

    # Data distribution - by SP groups
    total_files = len(all_eval_data)

    require_sample_padding = False
    
    if total_files == 0:
        logging.error(f"ERROR: No evaluation files found")
        this_rank_eval_data = []
    else:
        # Pad to match number of SP groups
        remainder = total_files % num_sp_groups
        if require_sample_padding and remainder != 0:
            pad_count = num_sp_groups - remainder
            all_eval_data += [all_eval_data[0]] * pad_count
        
        # Distribute across SP groups
        this_rank_eval_data = all_eval_data[sp_group_id :: num_sp_groups]

    # 为每个条件目录维护序列号计数器
    sequence_counters = {}
    skip_existing_outputs = get_config_bool(config, "skip_existing_outputs", False)
    saved_count = 0
    skipped_count = 0

    for data_idx, (text_prompt, image_path, ip_image_paths, ip_audio_paths) in tqdm(enumerate(this_rank_eval_data)):
        video_frame_height_width = config.get("video_frame_height_width", None)
        seed = config.get("seed", 100)
        solver_name = config.get("solver_name", "unipc")
        sample_steps = config.get("sample_steps", 50)
        shift = config.get("shift", 5.0)
        video_guidance_scale = config.get("video_guidance_scale", 4.0)
        audio_guidance_scale = config.get("audio_guidance_scale", 3.0)
        slg_layer = config.get("slg_layer", 11)
        fps = int(config.get("fps", 24))
        video_negative_prompt = config.get("video_negative_prompt", "")
        audio_negative_prompt = config.get("audio_negative_prompt", "")
        crop_face = getattr(ovi_engine, "crop_face", config.get("crop_face", False))

        # Validate IP image paths (one per reference person)
        ip_image_paths = [p for p in (ip_image_paths or []) if p is not None and os.path.isfile(p)]

        # Validate IP audio paths (one per reference person)
        ip_audio_paths = [p for p in (ip_audio_paths or []) if p is not None and os.path.isfile(p)]

        each_example_n_times = config.get("each_example_n_times", 1)
        ref_tag = f"_N{len(ip_image_paths)}" if len(ip_image_paths) > 1 else ""
        condition_dir = f"ip_image_{len(ip_image_paths) > 0}_ip_audio_{len(ip_audio_paths) > 0}{ref_tag}"
        condition_output_dir = os.path.join(output_dir, condition_dir)
        formatted_prompt = format_prompt_for_filename(text_prompt)
        save_rank = global_rank - sp_rank

        for idx in range(each_example_n_times):
            if skip_existing_outputs:
                sequence_number = data_idx * each_example_n_times + idx + 1
                sequence_str = f"{sequence_number:05d}"
                output_filename = f"{sequence_str}_crop-{crop_face}_{formatted_prompt}_{'x'.join(map(str, video_frame_height_width))}_{seed+idx}_{save_rank}.mp4"
                existing_output_path = find_existing_output(condition_output_dir, output_filename)
                if existing_output_path is not None:
                    skipped_count += 1
                    if sp_rank == 0:
                        logging.info(f"Skipping existing output: {existing_output_path}")
                    continue

            generated_video, generated_audio, generated_image = ovi_engine.generate(text_prompt=text_prompt,
                                                                    image_path=image_path,
                                                                    ip_image_paths=ip_image_paths,
                                                                    ip_audio_paths=ip_audio_paths,
                                                                    video_frame_height_width=video_frame_height_width,
                                                                    seed=seed+idx,
                                                                    solver_name=solver_name,
                                                                    sample_steps=sample_steps,
                                                                    shift=shift,
                                                                    video_guidance_scale=video_guidance_scale,
                                                                    audio_guidance_scale=audio_guidance_scale,
                                                                    slg_layer=slg_layer,
                                                                    video_negative_prompt=video_negative_prompt,
                                                                    audio_negative_prompt=audio_negative_prompt)
                                                                    
            if sp_rank == 0:
                # Create subdirectory based on input conditions
                os.makedirs(condition_output_dir, exist_ok=True)
                
                if skip_existing_outputs:
                    sequence_number = data_idx * each_example_n_times + idx + 1
                else:
                    # 获取或初始化该条件目录的序列号
                    if condition_dir not in sequence_counters:
                        sequence_counters[condition_dir] = get_next_sequence_number(output_dir, condition_dir)
                    
                    sequence_number = sequence_counters[condition_dir]
                    sequence_counters[condition_dir] += 1
                
                # 生成带有序号的文件名 (5位数字，前面补零)
                sequence_str = f"{sequence_number:05d}"
                output_filename = f"{sequence_str}_crop-{crop_face}_{formatted_prompt}_{'x'.join(map(str, video_frame_height_width))}_{seed+idx}_{global_rank}.mp4"
                output_path = os.path.join(condition_output_dir, output_filename)
                
                # Save video with audio
                save_video(output_path, generated_video, generated_audio, fps=fps, sample_rate=16000)
                logging.info(f"Saved video to: {output_path}")
                saved_count += 1
                
                # Save generated image if exists
                if generated_image is not None:
                    image_output_path = output_path.replace('.mp4', '.png')
                    generated_image.save(image_output_path)
                    logging.info(f"Saved image to: {image_output_path}")

    logging.info(f"Rank {global_rank} finished. Saved videos: {saved_count}, skipped existing videos: {skipped_count}")


if __name__ == "__main__":
    args = get_arguments()
    config = OmegaConf.load(args.config_file)
    main(config=config, args=args)
