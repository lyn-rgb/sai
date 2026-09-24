import atexit
import gc
import json
import os
import os.path as osp
import shutil
import tempfile

import math
import numpy as np
#import cv2
import pandas as pd
import librosa
import random
from copy import deepcopy

import torch  # NOTE, import torch before decord to avoid bug occurs in decord
import torchaudio
import decord
decord.bridge.set_bridge("torch")    # Read to torch.Tensor directly

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from . import video_transforms


# 语料里的视频常常没有扩展名（例如 `.../clips/<hash>`）
VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"}

_VIDEO_ALIAS_DIR = None
_VIDEO_ALIASES: dict = {}


def _video_alias(video_path: str) -> str:
    """给无扩展名的视频做一个带 `.mp4` 的软链接（部分 decord 版本按扩展名分派解码器）。"""
    global _VIDEO_ALIAS_DIR
    resolved = osp.abspath(video_path)
    alias = _VIDEO_ALIASES.get(resolved)
    if alias is None:
        if _VIDEO_ALIAS_DIR is None:
            _VIDEO_ALIAS_DIR = tempfile.mkdtemp(prefix="sai_video_alias_")
            atexit.register(shutil.rmtree, _VIDEO_ALIAS_DIR, ignore_errors=True)
        alias = osp.join(_VIDEO_ALIAS_DIR, f"{len(_VIDEO_ALIASES):06d}.mp4")
        os.symlink(resolved, alias)
        _VIDEO_ALIASES[resolved] = alias
    return alias


def outside_pieces(start: float, end: float, window_start: float, window_end: float) -> list:
    """`(start, end)` 落在目标窗口之外的部分（可能被窗口切成两段）。"""
    pieces = []
    if start < window_start:
        pieces.append((start, min(end, window_start)))
    if end > window_end:
        pieces.append((max(start, window_end), end))
    return [(a, b) for a, b in pieces if b > a]


def open_video_reader(video_path):
    """打开视频文件。

    无扩展名时先按原路径直接打开（ffmpeg 通常能按内容嗅探容器；注意**不能**把字节喂给 decord，
    有些版本会直接报 `Don't know how to handle type <class 'bytes'>`），失败再用带 `.mp4`
    后缀的软链接重试。
    """
    path = str(video_path)
    if osp.splitext(path)[-1].lower() in VIDEO_EXTS:
        return decord.VideoReader(path)
    try:
        return decord.VideoReader(path)
    except Exception:
        return decord.VideoReader(_video_alias(path))


class TextAudioVideoDataset(Dataset):
    """ NOTE, only supports batch size = 1 yet.
    """
    def __init__(
        self, 
        data_root, 
        meta_dir, 
        audio_sr=16000, 
        ref_audio_frames=48, 
        normalize_audio=True, 
        target_fps=24, 
        height=480, 
        width=864, 
        height_div=32, 
        width_div=32, 
        num_frames=81,
        min_video_len=10
    ):
        super().__init__()
        self.data_root = data_root
        
        self.audio_sr = audio_sr
        self.ref_audio_frames = ref_audio_frames
        self.normalize_audio = normalize_audio
        self.target_fps = target_fps
        self.num_pixels = height * width
        self.height_div = height_div
        self.width_div = width_div
        self.num_frames = num_frames
        self.min_video_frames = target_fps * min_video_len
        
        self.data = self._load_data(meta_dir)

    def _load_data(self, meta_dir):
        meta_names = [meta_name for meta_name in os.listdir(meta_dir) if meta_name.endswith("csv")]
        data = []
        for meta_name in meta_names:
            meta_path = osp.join(meta_dir, meta_name)
            pd_data = pd.read_csv(meta_path)
            pd_data = pd_data[pd_data["num_frames"] >= self.min_video_frames]
            data.extend([pd_data.iloc[i].to_dict() for i in range(len(pd_data))])
        return data

    @staticmethod
    def _opt(sample, key, default=None):
        """`pd.read_csv(...).to_dict()` turns missing cells into NaN, not None."""
        value = sample.get(key, None)
        if value is None:
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        if isinstance(value, str) and value.strip() == "":
            return default
        return value

    def get_transform(self, resize_height, resize_width, crop_height, crop_width):
        video_transform = transforms.Compose(
            [
                video_transforms.ToTensorVideo(),  # -> T,C,H,W, [0, 1]
                video_transforms.ResizeVideo((resize_height, resize_width), interpolation_mode="bilinear"),
                video_transforms.CenterCropVideo((crop_height, crop_width)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)  # [-1, 1], T,C,H,W
            ]
        )
        return video_transform
    
    def load_video(self, video_path, bbox=None, target_start_idx=None):
        video_reader = open_video_reader(video_path)
        source_fps = video_reader.get_avg_fps()
        frame_index_delta = source_fps / self.target_fps    # 48 / 24 = 2
        video_length = len(video_reader)                    # 240

        # random start frame index, left enthough length for reference audio
        total_target_frames = math.floor(video_length / frame_index_delta)    # 120
        if target_start_idx is None:
            assert total_target_frames >= (self.num_frames + self.ref_audio_frames), f"Require at least {(self.num_frames + self.ref_audio_frames)} frames, but video {video_path} only has {total_target_frames}"
            #if np.random.rand() < 0.5:
            target_start_idx = np.random.randint(self.ref_audio_frames, max(0, total_target_frames - self.num_frames))
            #else:
            #    target_start_idx = np.random.randint(0, max(0, total_target_frames - self.num_frames - self.ref_audio_frames))
        else:
            # fixed window, given by the annotation (`target_start_frame`). Rows that carry it take
            # their reference audio from outside the window, so no lead-in is required here.
            target_start_idx = int(target_start_idx)
            assert 0 <= target_start_idx and target_start_idx + self.num_frames <= total_target_frames, \
                f"Target window [{target_start_idx}, {target_start_idx + self.num_frames}) does not fit in {total_target_frames} frames of {video_path}"
        source_start_idx = int(target_start_idx * frame_index_delta)
        
        # calculate frame indices
        frame_ids = [source_start_idx]
        for i in range(1, self.num_frames):
            frame_id = source_start_idx + int(i * frame_index_delta)
            frame_ids.append(frame_id)
        
        frame_ids = np.array(frame_ids, dtype=np.int32)
        video = video_reader.get_batch(frame_ids).permute(0, 3, 1, 2)   # T,C,H,W
        
        # prcess video
        _, _, oh, ow = video.shape
        ratio = math.sqrt(self.num_pixels / (oh * ow))
        rh, rw = int(oh * ratio), int(ow * ratio)    # resize to
        ch = (rh // self.height_div) * self.height_div   # crop to
        cw = (rw // self.width_div) * self.width_div
        
        #print(f"video shape: ({oh}, {ow}), max number of pixels: {self.num_pixels}, ratio: {ratio}, resize: ({rh}, {rw}), crop size: ({ch}, {cw})")
        video = self.get_transform(rh, rw, ch, cw)(video)

        # process reference image
        # random a frame for ip_image
        ref_id = np.random.randint(0, video_length)
        ref_id = np.array([ref_id], dtype=np.int32)
        ref_image = video_reader.get_batch(ref_id).permute(0, 3, 1, 2)  # 1,C,H,W
        left, top, right, down = 0, 0, 1, 1
        if bbox is not None:
            left, top, right, down = bbox.split(',')
            left = float(left)
            top = float(top)
            right = float(right)
            down = float(down)
            
        _, _, ref_oh, ref_ow = ref_image.shape
        left, top  = max(0, int(left * ref_ow)), max(0, int(top * ref_oh))
        right, down = min(int(right * ref_ow), ref_ow), min(int(down * ref_oh), ref_oh)
        
        # crop image
        ref_image = ref_image[:, :, top:down, left:right]
        # resize and crop image
        _, _, ref_oh, ref_ow = ref_image.shape
        ref_rh, ref_rw = int(ref_oh * ratio), int(ref_ow * ratio)    # resize to
        if ref_rh * ref_rw < 256 * 256:                              # make sure reference image is large enougth
            ratio = math.sqrt(256 * 256 / (ref_oh * ref_ow))
            ref_rh, ref_rw = int(ref_oh * ratio), int(ref_ow * ratio)    # resize to
    
        ref_ch = (ref_rh // self.height_div) * self.height_div   # crop to
        ref_cw = (ref_rw // self.width_div) * self.width_div

        ref_image = self.get_transform(ref_rh, ref_rw, ref_ch, ref_cw)(ref_image)
        
        video = video.transpose(0, 1)
        ref_image = ref_image.transpose(0, 1)

        return video, ref_image, video_length, source_start_idx, source_fps

    def load_audio(self, audio_path, video_length, source_start_idx, source_fps, need_ref_audio=True):
        # NOTE, 使用torchaudio加载
        wave_data, sample_rate = torchaudio.load(str(audio_path))
        wave_data = torch.mean(wave_data, dim=0, keepdim=True)
        if sample_rate != self.audio_sr:
            wave_data = torchaudio.transforms.Resample(sample_rate, self.audio_sr)(wave_data)
        wave_data = wave_data[0]
        
        # NOTE, 使用librosa加载
        # wave_data, sample_rate = librosa.load(audio_path, sr=self.audio_sr, mono=True)
        # assert (sample_rate == self.audio_sr)
        #wave_data = torch.from_numpy(deepcopy(wave_data))
        
        # NOTE, [检查长度] 保证音频长度和视频长度差不超过7帧，7 * 160000 / 24
        audio_length = len(wave_data)
        audio_unit = self.audio_sr / self.target_fps    # number of samples per frame
        audio_num_frames = (audio_length / self.audio_sr) * self.target_fps
        video_num_frames = (video_length / source_fps) * self.target_fps
        assert abs(audio_num_frames - video_num_frames) < 8, f"{audio_path}, abs(audio_len - video_len)={abs(audio_num_frames - video_num_frames)} > 7 frames."
        
        # NOTE, [归一化], normalize
        max_wave = torch.max(torch.abs(wave_data))
        wave_data_normalized = wave_data / (max_wave + 1e-6) * 0.95   # align with mmaudio/data/extraction.wav_dataset.py line 89
                    
        # NOTE, [对齐音视频长度] align audio & video length
        aligned_audio_length = math.ceil((video_length / source_fps) * self.target_fps * audio_unit)  #向上取整，保证音频长度
        if aligned_audio_length > audio_length:
            # 0-padding
            wave_data = torch.cat([
                wave_data, 
                torch.zeros((aligned_audio_length - audio_length,), dtype=wave_data.dtype)
            ], dim=0)
            
            wave_data_normalized = torch.cat([
                wave_data_normalized, 
                torch.zeros((aligned_audio_length - audio_length,), dtype=wave_data_normalized.dtype)
            ], dim=0)
        
        # NOTE, [采样目标音频] sample audio
        tar_audio_length = int(self.num_frames * audio_unit)                   # 81 * 16000 / 24 = 54000 or 121 * 16000 / 24 = 80667
        audio_start_idx = int(source_start_idx * (self.audio_sr / source_fps))
        tar_audio = wave_data[audio_start_idx: audio_start_idx + tar_audio_length]
        tar_audio_normalized = wave_data_normalized[audio_start_idx: audio_start_idx + tar_audio_length]
        # padding if needed
        if len(tar_audio) < tar_audio_length:
            print(f"{audio_path}, padding audio from {tar_audio} to {tar_audio_length} samples")
            tar_audio = torch.cat([
                tar_audio,
                torch.zeros((tar_audio_length - len(tar_audio),), dtype=tar_audio.dtype)
            ], dim=0)
            
            tar_audio_normalized = torch.cat([
                tar_audio_normalized,
                torch.zeros((tar_audio_length - len(tar_audio),), dtype=tar_audio_normalized.dtype)
            ], dim=0)
        
        # NOTE, [采样参考音频] get reference audio
        ref_audio, ref_audio_normalized = None, None
        if need_ref_audio:
            # one random window of the target track, restricted to the region before the target clip
            ref_audio_length = int(self.ref_audio_frames * audio_unit)      # 2 * 24 * 16000 / 24 = 32000
            assert audio_start_idx - ref_audio_length >= 0, f"start index of audio {audio_start_idx} should larger than length of reference audio {ref_audio_length}"
            ref_audio_start_idx = np.random.randint(0, audio_start_idx - ref_audio_length + 1)   # avoid overlapping with tar_audio

            ref_audio = wave_data[ref_audio_start_idx: ref_audio_start_idx + ref_audio_length]
            ref_audio_normalized = wave_data_normalized[ref_audio_start_idx: ref_audio_start_idx + ref_audio_length]

        return tar_audio, tar_audio_normalized, ref_audio, ref_audio_normalized

    def sample_data(self, sample, need_ref_audio=True):
        video_path = osp.join(self.data_root, sample["video_path"])
        audio_path = osp.join(self.data_root, sample["audio_path"])
        bbox = self._opt(sample, "bbox")     # only used by the single-person `ip_image` path

        # `target_start_frame` (optional) pins the target window; multi-person rows use it so their
        # reference audio can be sliced strictly outside the window
        target_start_frame = self._opt(sample, "target_start_frame")
        target_start_frame = None if target_start_frame is None else int(target_start_frame)

        video, ref_image, video_length, source_start_idx, source_fps = self.load_video(video_path, bbox, target_start_frame)
        audio, audio_normalized, ref_audio, ref_audio_normalized = self.load_audio(audio_path, video_length, source_start_idx, source_fps, need_ref_audio=need_ref_audio)

        target_start_s = source_start_idx / source_fps
        target_window_s = (target_start_s, target_start_s + self.num_frames / self.target_fps)

        return video, ref_image, audio, audio_normalized, ref_audio, ref_audio_normalized, target_window_s
    
    def __getitem__(self, index):
        video, ref_image, audio, audio_normalized, \
        ref_audio, ref_audio_normalized, caption = None, None, None, None, None, None, None
        for i in range(100):
            try:
                sample = self.data[index]
                video, ref_image, audio, audio_normalized, ref_audio, ref_audio_normalized, _ = self.sample_data(sample)
                caption = sample["caption"]
                caption = caption.replace("<AUDCAP>", "Audio: ")   # to fit the 1.1 version
                caption = caption.replace("<ENDAUDCAP>", " ")
                break
            except Exception as e:
                index = random.randint(0, len(self.data) - 1)
                print(f"[{i}-th try Error] dataset, error <{e}> occurred when loading data from {sample['video_path']} and {sample['audio_path']}")
                
        return {
            "video": video,
            "audio": audio_normalized,
            "prompts": caption,
            "ip_image": ref_image,
            "ip_audio": ref_audio_normalized,
            "ori_audio": audio,
            "ori_ip_audio": ref_audio
        }

    def __len__(self):
        return len(self.data)


class TextAudioVideoFaceDataset(TextAudioVideoDataset):
    """ Reference-conditioned dataset supporting N reference persons per sample.

    Reference material is read from per-person columns written as semicolon separated lists
    (`face_paths`, `feat_paths`, `spk_audio_paths`); the position in the list defines the
    reference slot, i.e. slot 0 is the first person. Legacy single-person rows (`face_path` /
    `feat_path` only) are supported as well and are expanded to `n_refs` slots.

    Every sample returns exactly `n_refs` slots with fixed tensor shapes, so the default collate
    (`torch.stack`) keeps working. Slots without real reference material duplicate the first
    valid slot and are flagged in `ref_valid` (0.0).

    NOTE, the per-person reference audio (`spk_audio_paths` + `spk_segments`) is sliced from the
    segments that do NOT overlap the target window, otherwise the reference literally contains the
    prediction target. Rows that carry `target_start_frame` pin the target window so that this is
    well defined.
    """

    IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]

    def __init__(self, *args, n_refs=1, **kwargs):
        self.n_refs = n_refs
        super().__init__(*args, **kwargs)
        self.ref_audio_length = int(self.ref_audio_frames * self.audio_sr / self.target_fps)

    # ------------------------------------------------------------------ helpers
    def _person_paths(self, sample, list_key, single_key=None):
        """Semicolon separated per-person paths, falling back to a legacy single column."""
        raw = self._opt(sample, list_key)
        if raw is not None:
            paths = [p.strip() for p in str(raw).split(";")]
            assert all(p != "" for p in paths), \
                f"`{list_key}` must not contain empty entries (got {raw!r})"
            return paths
        single = self._opt(sample, single_key) if single_key is not None else None
        return [single] if single is not None else []

    def _person_path_groups(self, sample, list_key):
        """Two level path column: persons separated by `;`, files of one person by `,`."""
        raw = self._opt(sample, list_key)
        if raw is None:
            return []
        groups = []
        for group in str(raw).split(";"):
            paths = [p.strip() for p in group.split(",") if p.strip()]
            assert len(paths) > 0, f"`{list_key}` has an empty person group (got {raw!r})"
            groups.append(paths)
        return groups

    def _person_intervals(self, sample, key, n_persons):
        """Per-person list of `[start, end]` seconds, one entry per reference audio file."""
        raw = self._opt(sample, key)
        if raw is None:
            return [[] for _ in range(n_persons)]
        intervals = json.loads(raw)
        assert len(intervals) == n_persons, \
            f"`{key}` describes {len(intervals)} persons but the row has {n_persons}"
        return intervals

    def _load_wave(self, audio_path):
        wave_data, sample_rate = torchaudio.load(str(audio_path))
        wave_data = torch.mean(wave_data, dim=0, keepdim=True)
        if sample_rate != self.audio_sr:
            wave_data = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.audio_sr)(wave_data)
        return wave_data[0]

    def _fit_ref_audio(self, wave, normalize):
        """Pad/crop to the reference length, optionally peak normalized like `load_audio`."""
        if normalize:
            wave = wave / (torch.max(torch.abs(wave)) + 1e-6) * 0.95
        if len(wave) < self.ref_audio_length:
            wave = torch.cat([wave, wave.new_zeros(self.ref_audio_length - len(wave))])
        return wave[: self.ref_audio_length]

    # ------------------------------------------------------------------ references
    def load_ref_face(self, face_path, feat_path):
        """Load one person's reference face.

        Returns (face [3, 512, 512] in [-1, 1], face_emb [512]). `face_path` is either a still
        image (per-person reference photo) or a cropped face video (a random valid frame is used).
        `feat_path` holds the precomputed ArcFace features, either the per-frame format
        `{"face_embs": [T, 512], "valids": [T]}` or a single-image `{"face_emb": [512]}`.
        """
        # load reference face embedding
        feat = torch.load(feat_path)
        if "face_embs" in feat:
            valids = feat["valids"]
            valid_indices = torch.nonzero(valids)
            if valid_indices.numel() == 0:
                # no valid faces
                ref_idx = random.randint(0, valids.shape[0]-1)
            else:
                indices_list = valid_indices.tolist()  # 将张量转换为Python列表
                ref_idx = random.choice(indices_list)

            face_embs = feat["face_embs"]
            face_emb = face_embs[ref_idx].reshape(-1).to(torch.float32)
        else:
            ref_idx = 0
            face_emb = feat["face_emb"].reshape(-1).to(torch.float32)

        # load reference face frame
        if osp.splitext(str(face_path))[-1].lower() in self.IMAGE_EXTS:
            from torchvision.io import read_image
            face = read_image(str(face_path))                   # C,H,W, uint8
            if face.size(0) == 4:                               # drop alpha
                face = face[:3]
            elif face.size(0) == 1:                             # grayscale -> rgb
                face = face.expand(3, -1, -1)
            face = face.unsqueeze(0)                            # 1,C,H,W
        else:
            face_reader = open_video_reader(face_path)
            face = face_reader.get_batch([ref_idx]).permute(0, 3, 1, 2)  # 1,C,H,W
        face = self.get_transform(512, 512, 512, 512)(face)     # 1,3,512,512, [-1, 1]

        return face.squeeze(0), face_emb    # 3,512,512 and 512

    def load_ref_faces(self, sample):
        face_paths = self._person_paths(sample, "face_paths", "face_path")
        feat_paths = self._person_paths(sample, "feat_paths", "feat_path")
        assert len(face_paths) > 0, f"No reference face given for sample {sample.get('video_path')}"

        faces, face_embs = [], []
        for i, face_path in enumerate(face_paths[: self.n_refs]):
            feat_path = feat_paths[i] if i < len(feat_paths) else None
            assert feat_path is not None, \
                f"`feat_paths` must provide one feature file per reference face (sample {sample.get('video_path')})"
            face, face_emb = self.load_ref_face(osp.join(self.data_root, face_path),
                                                osp.join(self.data_root, feat_path))
            faces.append(face)
            face_embs.append(face_emb)
        return faces, face_embs

    def load_ref_audios(self, sample, target_window_s, ref_audio, ref_audio_normalized, spk_groups=None):
        """Per-person reference audio built from the segments outside the target window.

        `spk_groups` holds one person per `;` separated group with one file per `,` separated entry;
        `spk_segments` carries the matching `[start, end]` seconds so overlapping segments can be
        dropped. Without any per-person audio (legacy single-person rows) the random window sampled
        by `load_audio` is shared by every slot.
        """
        spk_groups = self._person_path_groups(sample, "spk_audio_paths") if spk_groups is None else spk_groups
        if len(spk_groups) == 0:
            # legacy behaviour: one random window of the target track, same for every slot
            return [ref_audio] * self.n_refs, [ref_audio_normalized] * self.n_refs

        intervals = self._person_intervals(sample, "spk_segments", len(spk_groups))
        target_start_s, target_end_s = target_window_s

        raws, normalized = [], []
        for person_idx, paths in enumerate(spk_groups[: self.n_refs]):
            spans = intervals[person_idx] if person_idx < len(intervals) else []
            waves, total = [], 0
            for file_idx, path in enumerate(paths):
                wave = self._load_wave(osp.join(self.data_root, path))
                span = spans[file_idx] if file_idx < len(spans) else None
                if span is None:
                    pieces = [wave]
                else:
                    # 只取落在目标窗口之外的部分：转换脚本按「窗口外语音总量」判定可行性，
                    # 这里若把「部分重叠」的整段丢掉，两边口径不一致，会误报参考音频不足
                    span_start, span_end = float(span[0]), float(span[1])
                    pieces = []
                    for piece_start, piece_end in outside_pieces(span_start, span_end, target_start_s, target_end_s):
                        begin = max(0, int(round((piece_start - span_start) * self.audio_sr)))
                        finish = min(len(wave), int(round((piece_end - span_start) * self.audio_sr)))
                        if finish > begin:
                            pieces.append(wave[begin:finish])
                for piece in pieces:
                    waves.append(piece)
                    total += len(piece)
                if total >= self.ref_audio_length:
                    break
            # 允许 2% 的取整误差（下面的 _fit_ref_audio 会补零到固定长度），真正的缺失仍会被拦住
            required = self.ref_audio_length * 0.98
            assert total >= required, (
                f"person {person_idx} of {sample.get('video_path')} has only {total / self.audio_sr:.2f}s of "
                f"reference audio outside the target window {target_window_s}, "
                f"{self.ref_audio_length / self.audio_sr:.2f}s required "
                f"(该样本的 spk_segments 与 target_start_frame 不匹配，需重建 meta CSV)")
            reference = torch.cat(waves)
            raws.append(self._fit_ref_audio(reference, normalize=False))
            normalized.append(self._fit_ref_audio(reference, normalize=True))
        return raws, normalized

    # ------------------------------------------------------------------ sampling
    def sample_data(self, sample):
        spk_groups = self._person_path_groups(sample, "spk_audio_paths")
        video, _, audio, audio_normalized, ref_audio, ref_audio_normalized, target_window_s = \
            super().sample_data(sample, need_ref_audio=len(spk_groups) == 0)

        faces, face_embs = self.load_ref_faces(sample)
        ref_audios, ref_audios_normalized = self.load_ref_audios(
            sample, target_window_s, ref_audio, ref_audio_normalized, spk_groups=spk_groups)

        # pad missing slots by duplicating the first reference (never zero-fill: a zero latent
        # frame would still occupy attention keys) and flag them in `ref_valid`
        ref_valid = [1.0] * len(faces)
        while len(faces) < self.n_refs:
            faces.append(faces[0])
            face_embs.append(face_embs[0])
            ref_audios.append(ref_audios[0])
            ref_audios_normalized.append(ref_audios_normalized[0])
            ref_valid.append(0.0)

        ip_image = torch.stack(faces, dim=1)                      # 3,N,512,512
        ip_image_embs = torch.stack(face_embs, dim=0)             # N,512
        ip_audio = torch.stack(ref_audios_normalized, dim=0)      # N,L (normalized)
        ori_ip_audio = torch.stack(ref_audios, dim=0)             # N,L (unnormalized)

        return video, ip_image, ip_image_embs, audio, audio_normalized, \
            ip_audio, ori_ip_audio, torch.tensor(ref_valid, dtype=torch.float32)

    def __getitem__(self, index):
        video, ip_image, ip_image_embs, audio, audio_normalized, \
        ip_audio, ori_ip_audio, ref_valid, caption = None, None, None, None, None, None, None, None, None
        for i in range(100):
            try:
                sample = self.data[index]
                video, ip_image, ip_image_embs, audio, audio_normalized, \
                ip_audio, ori_ip_audio, ref_valid = self.sample_data(sample)
                caption = sample["caption"]
                caption = caption.replace("<AUDCAP>", "Audio: ")   # to fit the 1.1 version
                caption = caption.replace("<ENDAUDCAP>", " ")
                break
            except Exception as e:
                index = random.randint(0, len(self.data) - 1)
                print(f"[{i}-th try Error] dataset, error <{e}> occurred when loading data from {sample['video_path']} and {sample['audio_path']}")

        return {
            "video": video,
            "audio": audio_normalized,
            "prompts": caption,
            "ip_image": ip_image,
            "ip_image_embs": ip_image_embs,
            "ip_audio": ip_audio,
            "ori_audio": audio,
            "ori_ip_audio": ori_ip_audio,
            "ref_valid": ref_valid,
        }
    