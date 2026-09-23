import os
import sys
import logging
import torch
import random
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


def is_existing_output_file(path):
    return os.path.isfile(path) and os.path.getsize(path) > 0


def find_existing_output(condition_output_dir, crop_face, text_prompt, video_frame_height_width, seed):
    if not os.path.isdir(condition_output_dir):
        return None

    formatted_prompt = format_prompt_for_filename(text_prompt)
    resolution = "x".join(map(str, video_frame_height_width))
    marker = f"_crop-{crop_face}_{formatted_prompt}_{resolution}_seed{seed}_"
    for filename in os.listdir(condition_output_dir):
        output_path = os.path.join(condition_output_dir, filename)
        if marker in filename and filename.endswith(".mp4") and is_existing_output_file(output_path):
            return output_path
    return None


def should_save_result(ip_image_path, ip_audio_path):
    """判断是否应该保存结果 - 硬编码规则"""
    has_ip_image = ip_image_path is not None and len(ip_image_path) > 0
    has_ip_audio = ip_audio_path is not None and len(ip_audio_path) > 0

    # 只保存同时有IP图像和IP音频的结果
    if has_ip_image and has_ip_audio:
        return True

    # 其他情况都不保存
    return False


def get_config_bool(config, key, default=False):
    value = config.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def parse_seed_list(seed_list):
    if seed_list is None:
        return None
    if isinstance(seed_list, str):
        seed_list = [seed.strip() for seed in seed_list.split(",") if seed.strip()]
    return [int(seed) for seed in seed_list]


def build_seed_list(config):
    configured_seed_list = parse_seed_list(config.get("seed_list", None))
    if configured_seed_list:
        return configured_seed_list

    base_seed = int(config.get("seed", 100))
    num_seeds = int(config.get("num_seeds", config.get("num_seeds_to_generate", 500)))
    if num_seeds <= 0:
        raise ValueError(f"num_seeds must be > 0, got {num_seeds}")

    if get_config_bool(config, "random_seeds", False):
        seed_min = int(config.get("random_seed_min", 1))
        seed_max = int(config.get("random_seed_max", 1000000))
        if seed_min > seed_max:
            raise ValueError(f"random_seed_min must be <= random_seed_max, got {seed_min} > {seed_max}")
        rng = random.Random(base_seed)
        return [rng.randint(seed_min, seed_max) for _ in range(num_seeds)]

    return [base_seed + i for i in range(num_seeds)]


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

    seed_list = build_seed_list(config)

    logging.info("=" * 60)
    logging.info(f"Generating {len(seed_list)} videos")
    if len(seed_list) <= 20:
        logging.info(f"Seeds: {seed_list}")
    else:
        logging.info(f"First 10 seeds: {seed_list[:10]}")
        logging.info(f"Last 10 seeds: {seed_list[-10:]}")
    logging.info("=" * 60)
    # ================================================

    # 打印保存规则（硬编码）
    logging.info("Save rules (hardcoded):")
    logging.info("  ✅ ONLY save results with BOTH IP image AND IP audio")
    logging.info("  ❌ Skip: no IP image & no IP audio")
    logging.info("  ❌ Skip: only IP image")
    logging.info("  ❌ Skip: only IP audio")
    logging.info("=" * 60)

    logging.info("Loading OVI Fusion Engine...")
    ovi_engine = OviFusionEngine(config=config, device=device, target_dtype=target_dtype, self_lora=self_lora)
    logging.info("OVI Fusion Engine loaded!")
    
    output_dir = config.get("output_dir", "./outputs")
    os.makedirs(output_dir, exist_ok=True)
    skip_existing_outputs = get_config_bool(config, "skip_existing_outputs", False)

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
    
    # 统计保存和跳过的数量
    saved_count = 0
    skipped_count = 0
    
    # 计算总生成任务数
    total_generation_tasks = len(this_rank_eval_data) * len(seed_list)
    logging.info(f"This rank will attempt to generate {total_generation_tasks} videos ({len(this_rank_eval_data)} prompts × {len(seed_list)} seeds)")

    for data_idx, (text_prompt, image_path, ip_image_paths, ip_audio_paths) in tqdm(enumerate(this_rank_eval_data), desc=f"Processing prompts (Rank {global_rank})"):
        video_frame_height_width = config.get("video_frame_height_width", None)
        solver_name = config.get("solver_name", "unipc")
        sample_steps = config.get("sample_steps", 50)
        shift = config.get("shift", 5.0)
        video_guidance_scale = config.get("video_guidance_scale", 4.0)
        audio_guidance_scale = config.get("audio_guidance_scale", 3.0)
        slg_layer = config.get("slg_layer", 11)
        fps = int(config.get("fps", 24))
        video_negative_prompt = config.get("video_negative_prompt", "")
        audio_negative_prompt = config.get("audio_negative_prompt", "")
        crop_face = config.get("crop_face", False)
        
        # Validate IP image paths (one per reference person)
        ip_image_paths = [p for p in (ip_image_paths or []) if p is not None]
        missing = [p for p in ip_image_paths if not os.path.isfile(p)]
        if len(missing) > 0:
            logging.warning(f"IP Images {missing} not exist, dropped")
            ip_image_paths = [p for p in ip_image_paths if os.path.isfile(p)]

        # Validate IP audio paths (one per reference person)
        ip_audio_paths = [p for p in (ip_audio_paths or []) if p is not None]
        missing = [p for p in ip_audio_paths if not os.path.isfile(p)]
        if len(missing) > 0:
            logging.warning(f"IP Audios {missing} not exist, dropped")
            ip_audio_paths = [p for p in ip_audio_paths if os.path.isfile(p)]

        # 检查是否需要保存这个条件的结果
        should_save = should_save_result(ip_image_paths, ip_audio_paths)

        # 记录当前条件的类型 (multi-person runs are kept in their own directory)
        has_ip_image = len(ip_image_paths) > 0
        has_ip_audio = len(ip_audio_paths) > 0
        ref_tag = f"_N{len(ip_image_paths)}" if len(ip_image_paths) > 1 else ""
        condition_type = f"ip_image_{has_ip_image}_ip_audio_{has_ip_audio}{ref_tag}"
        condition_dir = condition_type
        condition_output_dir = os.path.join(output_dir, condition_dir)
        
        if not should_save:
            # 计算这个条件会跳过的视频数量
            skip_count = len(seed_list)
            skipped_count += skip_count
            logging.info(f"⏭️  Skipping {skip_count} videos for condition: {condition_type}")
            continue
        
        logging.info(f"✅ Will generate {len(seed_list)} videos for condition: {condition_type}")
        
        # 为每个条件生成多个种子的结果
        for seed_idx, seed in enumerate(tqdm(seed_list, desc=f"Generating seeds for {condition_type}", leave=False)):
            try:
                if skip_existing_outputs:
                    existing_output_path = find_existing_output(
                        condition_output_dir,
                        crop_face,
                        text_prompt,
                        video_frame_height_width,
                        seed,
                    )
                    if existing_output_path:
                        skipped_count += 1
                        if sp_rank == 0:
                            logging.info(f"Skipping existing output for seed {seed}: {existing_output_path}")
                        continue

                logging.info(f"Generating video with seed {seed} for prompt: {text_prompt[:50]}...")
                
                generated_video, generated_audio, generated_image = ovi_engine.generate(
                    text_prompt=text_prompt,
                    image_path=image_path,
                    ip_image_paths=ip_image_paths,
                    ip_audio_paths=ip_audio_paths,
                    video_frame_height_width=video_frame_height_width,
                    seed=seed,
                    solver_name=solver_name,
                    sample_steps=sample_steps,
                    shift=shift,
                    video_guidance_scale=video_guidance_scale,
                    audio_guidance_scale=audio_guidance_scale,
                    slg_layer=slg_layer,
                    video_negative_prompt=video_negative_prompt,
                    audio_negative_prompt=audio_negative_prompt
                )
                
                if sp_rank == 0:
                    # Create subdirectory based on input conditions
                    os.makedirs(condition_output_dir, exist_ok=True)
                    
                    # 获取或初始化该条件目录的序列号
                    if condition_dir not in sequence_counters:
                        sequence_counters[condition_dir] = get_next_sequence_number(output_dir, condition_dir)
                    
                    sequence_number = sequence_counters[condition_dir]
                    sequence_counters[condition_dir] += 1
                    
                    formatted_prompt = format_prompt_for_filename(text_prompt)
                    
                    # 生成带有序号的文件名
                    sequence_str = f"{sequence_number:05d}"
                    output_filename = f"{sequence_str}_crop-{crop_face}_{formatted_prompt}_{'x'.join(map(str, video_frame_height_width))}_seed{seed}_{global_rank}.mp4"
                    output_path = os.path.join(condition_output_dir, output_filename)
                    
                    # Save video with audio
                    save_video(output_path, generated_video, generated_audio, fps=fps, sample_rate=16000)
                    logging.info(f"✅ Saved video to: {output_path}")
                    saved_count += 1
                    
                    # Save generated image if exists
                    if generated_image is not None:
                        image_output_path = output_path.replace('.mp4', '.png')
                        generated_image.save(image_output_path)
                        logging.info(f"✅ Saved image to: {image_output_path}")
                        
            except Exception as e:
                logging.error(f"❌ Error generating video for seed {seed}: {e}")
                logging.error(f"Error details: {str(e)}")
                continue
    
    # 打印统计信息
    logging.info(f"=" * 60)
    logging.info(f"Rank {global_rank} finished!")
    logging.info(f"✅ Saved videos: {saved_count}")
    logging.info(f"⏭️  Skipped videos: {skipped_count}")
    logging.info(f"📊 Total attempted: {saved_count + skipped_count}")
    logging.info(f"=" * 60)


if __name__ == "__main__":
    # 使用原有的参数解析函数
    args = get_arguments()
    
    # 加载配置
    config = OmegaConf.load(args.config_file)
    
    # 打印运行信息
    print("=" * 80)
    print("OVI Fusion Video Generation")
    print("=" * 80)
    print(f"Config file: {args.config_file}")
    
    seed_list = build_seed_list(config)
    print(f"Generating {len(seed_list)} videos")
    if len(seed_list) <= 20:
        print(f"Seeds: {seed_list}")
    else:
        print(f"First 10 seeds: {seed_list[:10]}")
        print(f"Last 10 seeds: {seed_list[-10:]}")
    
    print(f"Output directory: {config.get('output_dir', './outputs')}")
    print("\nSave rules (hardcoded):")
    print("  ✅ ONLY save results with BOTH IP image AND IP audio")
    print("  ❌ Skip: no IP image & no IP audio")
    print("  ❌ Skip: only IP image")
    print("  ❌ Skip: only IP audio")
    print("=" * 80)
    
    main(config=config, args=args)
