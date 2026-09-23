from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".gif"}
AUDIO_EXTS = {".wav", ".flac", ".mp3"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


class NullProgress:
    def __init__(self, iterable=None, total=None, desc=None, unit=None, **kwargs):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable or [])

    def set_description(self, desc: str) -> None:
        print(desc)

    def update(self, n: int = 1) -> None:
        pass

    def close(self) -> None:
        pass


def progress_iter(iterable, **kwargs):
    if tqdm is None:
        return NullProgress(iterable, **kwargs)
    return tqdm(iterable, dynamic_ncols=True, **kwargs)


def progress_bar(total: int, **kwargs):
    if tqdm is None:
        return NullProgress(total=total, **kwargs)
    return tqdm(total=total, dynamic_ncols=True, **kwargs)


def load_config(path: Path) -> dict[str, Any]:
    try:
        from omegaconf import OmegaConf
    except Exception as exc:
        raise RuntimeError("OmegaConf is required to read the evaluation YAML config.") from exc
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def apply_runtime_overrides(
    cfg: dict[str, Any],
    root: Path,
    output_dir: str | None = None,
    workspace_dir: str | None = None,
    no_skip_completed: bool = False,
) -> None:
    paths = cfg.setdefault("paths", {})
    if output_dir:
        resolved_output = as_path(output_dir, root)
        assert resolved_output is not None
        paths["output_dir"] = str(resolved_output)
        if not workspace_dir:
            paths["workspace_dir"] = str(resolved_output / "workspace")
    if workspace_dir:
        resolved_workspace = as_path(workspace_dir, root)
        assert resolved_workspace is not None
        paths["workspace_dir"] = str(resolved_workspace)
    if no_skip_completed:
        resume_cfg = cfg.setdefault("resume", {})
        resume_cfg["enabled"] = False
        resume_cfg["skip_completed_metrics"] = False


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def as_path(value: str | Path | None, root: Path) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else (root / path)


def metric_python_cmd(cfg: dict[str, Any], metric_name: str, python_bin: str) -> list[str]:
    metric_envs = cfg.get("metric_envs", {})
    env_name = metric_envs.get(metric_name)
    if not env_name:
        return [python_bin]
    conda_cfg = cfg.get("conda", {})
    conda_exe = conda_cfg.get("executable", "conda")
    return [conda_exe, "run", "--no-capture-output", "-n", str(env_name), "python"]


def run_cmd(
    cmd: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    log_path: Path | None = None,
) -> subprocess.CompletedProcess:
    print(" ".join(cmd))
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(result.stdout or "", encoding="utf-8")
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd, output=result.stdout)
    return result


def sample_ids_from_prompt_dir(prompt_dir: Path) -> list[str]:
    sample_ids = []
    for path in sorted(prompt_dir.glob("*_full_caption.txt")):
        sample_ids.append(path.name.removesuffix("_full_caption.txt"))
    return sample_ids


def collect_videos(path: Path) -> list[Path]:
    if not path.exists():
        return []
    if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
        return [path]
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS)


def infer_video_id(video_path: Path) -> str | None:
    stem = video_path.stem
    if re.match(r"^\d{5}(?:_|$)", stem):
        return stem[:5]
    match = re.search(r"(?<!\d)(\d{5})(?!\d)", stem)
    return match.group(1) if match else None


def link_or_copy(src: Path, dst: Path, copy_files: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_files:
        shutil.copy2(src, dst)
    else:
        try:
            os.symlink(src.resolve(), dst)
        except OSError:
            shutil.copy2(src, dst)


def map_videos_by_id(result_dir: Path, sample_ids: list[str]) -> tuple[list[Path], dict[str, Path]]:
    videos = collect_videos(result_dir)
    by_id: dict[str, Path] = {}
    for video in videos:
        sid = infer_video_id(video)
        if sid and sid in sample_ids and sid not in by_id:
            by_id[sid] = video

    missing_ids = [sid for sid in sample_ids if sid not in by_id]
    if missing_ids and len(videos) == len(sample_ids):
        for sid, video in zip(sample_ids, videos):
            by_id.setdefault(sid, video)
    return videos, by_id


def prepare_video_workspace(result_dir: Path, sample_ids: list[str], workspace: Path) -> tuple[Path, dict[str, str], list[str]]:
    videos, by_id = map_videos_by_id(result_dir, sample_ids)
    flat_video_dir = workspace / "videos"
    flat_video_dir.mkdir(parents=True, exist_ok=True)
    for old in flat_video_dir.iterdir():
        if old.is_file() or old.is_symlink():
            old.unlink()

    matched_ids = [sid for sid in sample_ids if sid in by_id]
    for sid in sample_ids:
        if sid in by_id:
            link_or_copy(by_id[sid], flat_video_dir / f"{sid}{by_id[sid].suffix.lower()}")

    meta = {
        "source_videos": str(len(videos)),
        "matched_videos": str(len(by_id)),
        "missing_videos": str(len([sid for sid in sample_ids if sid not in by_id])),
    }
    return flat_video_dir, meta, matched_ids


def prepare_video_subset_workspace(
    source_dir: Path,
    all_sample_ids: list[str],
    subset_ids: list[str],
    workspace: Path,
) -> Path:
    _, by_id = map_videos_by_id(source_dir, all_sample_ids)
    workspace.mkdir(parents=True, exist_ok=True)
    for old in workspace.iterdir():
        if old.is_file() or old.is_symlink():
            old.unlink()
    for sid in subset_ids:
        src = by_id.get(sid)
        if src is not None:
            link_or_copy(src, workspace / f"{sid}{src.suffix.lower()}")
    return workspace


def collect_files(path: Path, exts: set[str]) -> list[Path]:
    if not path.exists():
        return []
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def prepare_file_subset_workspace(
    source_dir: Path,
    all_sample_ids: list[str],
    subset_ids: list[str],
    workspace: Path,
    exts: set[str],
    output_name,
) -> Path:
    files = collect_files(source_dir, exts)
    by_id: dict[str, Path] = {}
    for path in files:
        sid = infer_video_id(path)
        if sid and sid in all_sample_ids and sid not in by_id:
            by_id[sid] = path
    missing_ids = [sid for sid in all_sample_ids if sid not in by_id]
    if missing_ids and len(files) == len(all_sample_ids):
        for sid, path in zip(all_sample_ids, files):
            by_id.setdefault(sid, path)

    workspace.mkdir(parents=True, exist_ok=True)
    for old in workspace.iterdir():
        if old.is_file() or old.is_symlink():
            old.unlink()
    for sid in subset_ids:
        src = by_id.get(sid)
        if src is not None:
            link_or_copy(src, workspace / output_name(sid, src))
    return workspace


def video_match_meta(result_dir: Path, sample_ids: list[str]) -> dict[str, str]:
    videos, by_id = map_videos_by_id(result_dir, sample_ids)
    return {
        "source_videos": str(len(videos)),
        "matched_videos": str(len(by_id)),
        "missing_videos": str(len([sid for sid in sample_ids if sid not in by_id])),
    }


def extract_audio(
    video_dir: Path,
    audio_dir: Path,
    ffmpeg_bin: str,
    sample_rate: int = 16000,
    desc: str = "Extract audio",
) -> None:
    audio_dir.mkdir(parents=True, exist_ok=True)
    for old in audio_dir.glob("*.wav"):
        old.unlink()
    videos = sorted(collect_videos(video_dir))
    for video in progress_iter(videos, desc=desc, unit="video", leave=False):
        sid = infer_video_id(video) or video.stem
        out = audio_dir / f"{sid}_audio.wav"
        cmd = [
            ffmpeg_bin,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            str(out),
        ]
        subprocess.run(cmd, check=True)


def prepare_wer_gt(asr_dir: Path, output_dir: Path, sample_ids: list[str]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.glob("*.txt"):
        old.unlink()
    for sid in sample_ids:
        src = asr_dir / f"{sid}_asr.txt"
        if src.is_file():
            (output_dir / f"{sid}.txt").write_text(src.read_text(encoding="utf-8").strip(), encoding="utf-8")
    return output_dir


def parse_stat_csv(path: Path, value_col: int = 1) -> dict[str, float]:
    stats = {}
    if not path.exists():
        return stats
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    for row in rows:
        if len(row) > value_col and row[0]:
            try:
                stats[row[0]] = float(row[value_col])
            except ValueError:
                pass
    return stats


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, variance ** 0.5


def merge_facesim_csv(partial_csvs: list[Path], output_csv: Path) -> dict[str, float]:
    header = ["prompt_id", "video_path", "image_path", "cur_score", "arc_score", "fid_score"]
    detail_rows: list[list[str]] = []
    for partial_csv in partial_csvs:
        if not partial_csv.exists():
            continue
        with partial_csv.open(newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if not row:
                    continue
                if row[0] == "STATISTICS":
                    break
                if row[0] == "prompt_id":
                    continue
                if len(row) >= 6:
                    detail_rows.append(row[:6])

    def sort_key(row: list[str]) -> tuple[int, str]:
        try:
            return int(row[0]), row[0]
        except ValueError:
            return sys.maxsize, row[0]

    detail_rows.sort(key=sort_key)
    scores = {"cur": [], "arc": [], "fid": []}
    valid_rows = []
    for row in detail_rows:
        try:
            cur_score = float(row[3])
            arc_score = float(row[4])
            fid_score = float(row[5])
        except ValueError:
            continue
        valid_rows.append(row)
        scores["cur"].append(cur_score)
        scores["arc"].append(arc_score)
        scores["fid"].append(fid_score)
    if not valid_rows:
        raise RuntimeError("No valid FaceSim rows were produced by the completed shards.")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(valid_rows)
        writer.writerow(["STATISTICS", "", "", "", "", ""])
        writer.writerow(["count", len(valid_rows), "", "", "", ""])
        for prefix in ["cur", "arc", "fid"]:
            mean, _ = mean_std(scores[prefix])
            writer.writerow([f"{prefix}_mean", f"{mean:.4f}", "", "", "", ""])
        for prefix in ["cur", "arc", "fid"]:
            _, std = mean_std(scores[prefix])
            writer.writerow([f"{prefix}_std", f"{std:.4f}", "", "", "", ""])
    return parse_stat_csv(output_csv)


def device_list(metric_cfg: dict[str, Any], default_device: str) -> list[str]:
    devices = metric_cfg.get("devices")
    if devices is None:
        return [metric_cfg.get("device", default_device)]
    if isinstance(devices, str):
        return [item.strip() for item in devices.split(",") if item.strip()]
    return [str(item) for item in devices if str(item).strip()]


def tee_stream(stream, log_handle, mirror: bool = False) -> None:
    while True:
        chunk = stream.read(1)
        if not chunk:
            break
        log_handle.write(chunk)
        log_handle.flush()
        if mirror:
            sys.stderr.write(chunk)
            sys.stderr.flush()


def run_cmd_tee(
    cmd: list[str],
    cwd: Path | None = None,
    log_path: Path | None = None,
) -> subprocess.CompletedProcess:
    print(" ".join(cmd))
    log_handle = None
    thread = None
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    try:
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            text=True,
            bufsize=1,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if log_handle is not None and process.stdout is not None:
            thread = threading.Thread(target=tee_stream, args=(process.stdout, log_handle, True), daemon=True)
            thread.start()
        elif process.stdout is not None:
            thread = threading.Thread(target=tee_stream, args=(process.stdout, sys.stderr, False), daemon=True)
            thread.start()
        return_code = process.wait()
        if thread is not None:
            thread.join()
        output = log_path.read_text(encoding="utf-8", errors="replace") if log_path is not None and log_path.exists() else ""
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd, output=output)
        return subprocess.CompletedProcess(cmd, return_code, stdout=output, stderr=None)
    finally:
        if log_handle is not None and not log_handle.closed:
            log_handle.close()


def run_facesim_shards(
    base_cmd: list[str],
    cwd: Path,
    face_csv: Path,
    log_dir: Path,
    parallel_jobs: int,
    devices: list[str],
) -> dict[str, float]:
    if not devices:
        raise RuntimeError("metrics.facesim.devices is empty.")

    if parallel_jobs <= 1:
        run_cmd_tee(
            base_cmd + ["--device", devices[0], "--results_csv", str(face_csv), "--overwrite"],
            cwd=cwd,
            log_path=log_dir / "facesim.log",
        )
        return parse_stat_csv(face_csv)

    partial_csvs = [face_csv.with_name(f"{face_csv.stem}_shard_{idx}.csv") for idx in range(parallel_jobs)]
    processes = []
    log_handles = []
    stream_threads: dict[int, threading.Thread] = {}
    for idx, partial_csv in enumerate(partial_csvs):
        if partial_csv.exists():
            partial_csv.unlink()
        cmd = base_cmd + [
            "--device",
            devices[idx % len(devices)],
            "--results_csv",
            str(partial_csv),
            "--overwrite",
            "--shard-index",
            str(idx),
            "--num-shards",
            str(parallel_jobs),
        ]
        log_path = log_dir / f"facesim_shard_{idx}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("w", encoding="utf-8")
        print(" ".join(cmd))
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        mirror_output = idx == 0
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            text=True,
            bufsize=1,
            stdout=subprocess.PIPE if mirror_output else log_handle,
            stderr=subprocess.STDOUT,
        )
        if mirror_output and process.stdout is not None:
            stream_thread = threading.Thread(target=tee_stream, args=(process.stdout, log_handle, True), daemon=True)
            stream_thread.start()
            stream_threads[idx] = stream_thread
        processes.append((idx, process, log_path))
        log_handles.append(log_handle)

    errors = []
    pending = {idx for idx, _, _ in processes}
    try:
        while pending:
            for idx, process, log_path in processes:
                if idx not in pending:
                    continue
                return_code = process.poll()
                if return_code is None:
                    continue
                pending.remove(idx)
                if idx in stream_threads:
                    stream_threads[idx].join()
                log_handles[idx].close()
                if return_code != 0:
                    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
                    errors.append(f"shard {idx} failed: {short_error(log_text)}")
            if pending:
                time.sleep(1)
    finally:
        for idx in list(pending):
            processes[idx][1].wait()
            if idx in stream_threads:
                stream_threads[idx].join()
            log_handles[idx].close()
        for handle in log_handles:
            if not handle.closed:
                handle.close()

    completed_csvs = [path for idx, path in enumerate(partial_csvs) if processes[idx][1].returncode == 0 and path.exists()]
    if errors and not completed_csvs:
        raise subprocess.CalledProcessError(1, base_cmd, output="\n".join(errors))
    stats = merge_facesim_csv(completed_csvs, face_csv)
    if errors:
        raise subprocess.CalledProcessError(1, base_cmd, output="\n".join(errors))
    return stats


def parse_wer_json(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    robust = data.get("statistics", {}).get("robust_metrics", {})
    basic = data.get("statistics", {})
    source = robust or basic
    mapping = {
        "wer": source.get("overall_wer", source.get("average_wer")),
        "wer_pct": source.get("overall_wer_percentage", source.get("wer_percentage")),
        "wer_count": source.get("normal_files_count", data.get("config", {}).get("total_files")),
    }
    return {k: float(v) for k, v in mapping.items() if v is not None}


def parse_audio_sim(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Total pairs:"):
            out["audio_sim_count"] = float(line.split(":", 1)[1].strip())
        elif line.startswith("Average similarity:"):
            out["audio_sim"] = float(line.split(":", 1)[1].strip())
        elif line.startswith("Std similarity:"):
            out["audio_sim_std"] = float(line.split(":", 1)[1].strip())
    return out


def parse_syncnet_json(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data.get("summary", {})
    values = {
        "sync_c": summary.get("average_confidence"),
        "sync_c_std": summary.get("std_confidence"),
        "sync_d": summary.get("average_distance"),
        "sync_d_std": summary.get("std_distance"),
        "sync_offset": summary.get("average_offset"),
        "sync_count": summary.get("successful_videos"),
        "sync_failed": summary.get("failed_videos"),
        "syncnet_preprocess_version": summary.get("preprocess_version"),
        "syncnet_use_crop_pipeline": summary.get("use_crop_pipeline"),
    }
    return {key: float(value) for key, value in values.items() if value is not None}


def load_metric_cache(detail_dir: Path) -> dict[str, dict[str, Any]]:
    cache_path = detail_dir / "metrics_cache.json"
    if not cache_path.exists():
        return {}
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def save_metric_cache(detail_dir: Path, cache: dict[str, dict[str, Any]]) -> None:
    cache_path = detail_dir / "metrics_cache.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def cache_metric_result(
    detail_dir: Path,
    cache: dict[str, dict[str, Any]],
    metric_name: str,
    values: dict[str, Any],
) -> None:
    clean = {key: jsonable(value) for key, value in values.items() if value is not None and not key.endswith("_error")}
    if not clean:
        return
    clean["_completed"] = True
    clean["_updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    cache[metric_name] = clean
    save_metric_cache(detail_dir, cache)


def cached_metric_result(metric_name: str, detail_dir: Path, cache: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    cached = cache.get(metric_name)
    if isinstance(cached, dict) and cached.get("_completed"):
        return {key: value for key, value in cached.items() if not key.startswith("_")}

    values: dict[str, Any] = {}
    if metric_name == "clip":
        stats = parse_stat_csv(detail_dir / "clip.csv")
        if stats.get("mean") is not None:
            values = {"clip_score": stats.get("mean"), "clip_std": stats.get("std"), "clip_count": stats.get("count")}
    elif metric_name == "facesim":
        stats = parse_stat_csv(detail_dir / "facesim.csv")
        if stats.get("cur_mean") is not None:
            values = {
                "facesim_cur": stats.get("cur_mean"),
                "facesim_arc": stats.get("arc_mean"),
                "facesim_fid": stats.get("fid_mean"),
                "facesim_count": stats.get("count"),
            }
    elif metric_name == "wer":
        corrected = detail_dir / "wer_results_corrected.json"
        raw = detail_dir / "wer_results.json"
        values = parse_wer_json(corrected if corrected.exists() else raw)
    elif metric_name == "audio_sim":
        values = parse_audio_sim(detail_dir / "audio_sim.txt")
    elif metric_name == "fad":
        fad_json = detail_dir / "fad.json"
        if fad_json.exists():
            values = json.loads(fad_json.read_text(encoding="utf-8"))
    elif metric_name == "fvd":
        fvd_json = detail_dir / "fvd.json"
        if fvd_json.exists():
            values = json.loads(fvd_json.read_text(encoding="utf-8"))
    elif metric_name == "syncnet":
        values = parse_syncnet_json(detail_dir / "syncnet.json")

    return values or None


def use_cached_metric(
    metric_name: str,
    row: dict[str, Any],
    detail_dir: Path,
    cache: dict[str, dict[str, Any]],
    skip_completed: bool,
    expected_count: int | None = None,
) -> bool:
    if not skip_completed:
        return False
    values = cached_metric_result(metric_name, detail_dir, cache)
    if not values:
        return False
    count_keys = {
        "clip": "clip_count",
        "wer": "wer_count",
        "audio_sim": "audio_sim_count",
        "fad": "fad_count",
        "fvd": "fvd_count",
        "syncnet": "sync_count",
    }
    count_key = count_keys.get(metric_name)
    if expected_count is not None and count_key:
        try:
            cached_count = int(float(values[count_key]))
        except (KeyError, TypeError, ValueError):
            print(f"Recompute {metric_name}: cached result has no compatible {count_key}.")
            return False
        if cached_count != expected_count:
            print(f"Recompute {metric_name}: cached count {cached_count} != matched count {expected_count}.")
            return False
    required_keys = {
        "syncnet": ["sync_c", "sync_d", "syncnet_preprocess_version"],
    }
    missing_required = [key for key in required_keys.get(metric_name, []) if key not in values]
    if missing_required:
        print(f"Recompute {metric_name}: cached result is missing {', '.join(missing_required)}.")
        return False
    if metric_name == "syncnet" and int(float(values["syncnet_preprocess_version"])) != 3:
        print("Recompute syncnet: cached result was produced by an old preprocessing version.")
        return False
    row.update(values)
    cache_metric_result(detail_dir, cache, metric_name, values)
    print(f"Skip {metric_name}: found completed result.")
    return True


def short_error(text: str, limit: int = 240) -> str:
    clean = " ".join(str(text).split())
    return clean[-limit:] if len(clean) > limit else clean


def md_cell(value: Any, limit: int = 240) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r", " ").replace("\n", "<br>")
    text = text.replace("|", "\\|")
    return text[:limit] + "..." if len(text) > limit else text


def write_table(rows: list[dict[str, Any]], output_csv: Path, output_md: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    keys = ["name", "result_dir", "matched_videos", "missing_videos", "error"]
    metric_keys = sorted({k for row in rows for k in row if k not in keys})
    fieldnames = keys + metric_keys
    tmp_csv = output_csv.with_suffix(output_csv.suffix + ".tmp")
    tmp_md = output_md.with_suffix(output_md.suffix + ".tmp")
    with tmp_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_csv.replace(output_csv)

    lines = ["| " + " | ".join(fieldnames) + " |", "| " + " | ".join(["---"] * len(fieldnames)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(md_cell(row.get(k, "")) for k in fieldnames) + " |")
    tmp_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp_md.replace(output_md)


def evaluate_one(
    name: str,
    result_dir: Path,
    cfg: dict[str, Any],
    root: Path,
    python_bin: str,
    only_metrics: set[str] | None = None,
) -> dict[str, Any]:
    paths = cfg["paths"]
    weights = cfg.get("weights", {})
    metrics = cfg.get("metrics", {})
    output_root = as_path(paths["output_dir"], root)
    workspace_root = as_path(paths.get("workspace_dir", str(output_root / "workspace")), root)
    testdata_dir = as_path(paths["testdata_dir"], root)
    assert output_root is not None and workspace_root is not None and testdata_dir is not None

    prompt_dir = testdata_dir / paths.get("prompt_dir_name", "full_video_prompt")
    frame_dir = testdata_dir / paths.get("frame_dir_name", "frames")
    audio_ref_dir = testdata_dir / paths.get("audio_dir_name", "audio")
    asr_dir = testdata_dir / paths.get("asr_dir_name", "ASR")
    sample_ids = sample_ids_from_prompt_dir(prompt_dir)

    dataset_workspace = workspace_root / name
    detail_dir = output_root / "details" / name
    log_dir = detail_dir / "logs"
    detail_dir.mkdir(parents=True, exist_ok=True)
    row: dict[str, Any] = {"name": name, "result_dir": str(result_dir), "error": ""}
    metric_cache = load_metric_cache(detail_dir)
    resume_cfg = cfg.get("resume", {})
    skip_completed = bool(resume_cfg.get("enabled", False) and resume_cfg.get("skip_completed_metrics", True))

    def metric_enabled(metric_name: str, default_enabled: bool = False) -> bool:
        if only_metrics and metric_name not in only_metrics:
            return False
        return bool(metrics.get(metric_name, {}).get("enabled", default_enabled))

    flat_video_dir, meta, matched_sample_ids = prepare_video_workspace(result_dir, sample_ids, dataset_workspace)
    row.update(meta)
    row["evaluated_videos"] = str(len(matched_sample_ids))
    if not matched_sample_ids:
        row["error"] = "no matched videos"
        return row

    prompt_eval_dir = prepare_file_subset_workspace(
        prompt_dir,
        sample_ids,
        matched_sample_ids,
        dataset_workspace / "prompts",
        {".txt"},
        lambda sid, src: f"{sid}_full_caption.txt",
    )
    frame_eval_dir = prepare_file_subset_workspace(
        frame_dir,
        sample_ids,
        matched_sample_ids,
        dataset_workspace / "frames",
        IMAGE_EXTS,
        lambda sid, src: f"{sid}_frame{src.suffix.lower()}",
    )
    audio_ref_eval_dir = prepare_file_subset_workspace(
        audio_ref_dir,
        sample_ids,
        matched_sample_ids,
        dataset_workspace / "ref_audio",
        AUDIO_EXTS,
        lambda sid, src: f"{sid}_audio{src.suffix.lower()}",
    )
    audio_dir = dataset_workspace / "audio"
    audio_metrics_to_run = [
        key
        for key in ["fad", "wer", "audio_sim"]
        if metric_enabled(key, False)
    ]
    if audio_metrics_to_run:
        extract_audio(flat_video_dir, audio_dir, cfg.get("ffmpeg_bin", "ffmpeg"), desc=f"{name}: extract audio")

    enabled_metrics = [
        metric_name
        for metric_name, default_enabled in [
            ("clip", True),
            ("facesim", True),
            ("wer", True),
            ("audio_sim", False),
            ("fad", False),
            ("fvd", False),
            ("syncnet", False),
        ]
        if metric_enabled(metric_name, default_enabled)
    ]
    metric_bar = progress_bar(len(enabled_metrics), desc=f"{name}: metrics", unit="metric", leave=False)

    if metric_enabled("clip", True):
        metric_bar.set_description(f"{name}: clip")
        try:
            clip_csv = detail_dir / "clip.csv"
            if not use_cached_metric("clip", row, detail_dir, metric_cache, skip_completed, len(matched_sample_ids)):
                run_cmd_tee(
                    [
                        *metric_python_cmd(cfg, "clip", python_bin),
                        "batch_clipscore.py",
                        "--video_root",
                        str(flat_video_dir),
                        "--prompt_root",
                        str(prompt_eval_dir),
                        "--model_path",
                        str(as_path(weights.get("clip_model_path"), root)),
                        "--device",
                        metrics["clip"].get("device", cfg.get("device", "cuda")),
                        "--num_frames",
                        str(metrics["clip"].get("num_frames", 16)),
                        "--results_csv",
                        str(clip_csv),
                        "--overwrite",
                    ],
                    cwd=root / "evaluation/codes/vision/clip",
                    log_path=log_dir / "clip.log",
                )
                stats = parse_stat_csv(clip_csv)
                values = {"clip_score": stats.get("mean"), "clip_std": stats.get("std"), "clip_count": stats.get("count")}
                row.update(values)
                cache_metric_result(detail_dir, metric_cache, "clip", values)
        except subprocess.CalledProcessError as exc:
            row["clip_error"] = short_error(exc.output or str(exc))
        except Exception as exc:
            row["clip_error"] = short_error(str(exc))
        finally:
            metric_bar.update(1)

    if metric_enabled("facesim", True):
        metric_bar.set_description(f"{name}: facesim")
        try:
            face_csv = detail_dir / "facesim.csv"
            if not use_cached_metric("facesim", row, detail_dir, metric_cache, skip_completed):
                facesim_cfg = metrics["facesim"]
                facesim_device = facesim_cfg.get("device", cfg.get("device", "cuda"))
                facesim_jobs = int(facesim_cfg.get("parallel_jobs", 1))
                stats = run_facesim_shards(
                    [
                        *metric_python_cmd(cfg, "facesim", python_bin),
                        "eval_batch_facesim_fid.py",
                        "--video_root",
                        str(flat_video_dir),
                        "--image_root",
                        str(frame_eval_dir),
                        "--face_model_path",
                        str(as_path(weights.get("face_model_path"), root)),
                        "--inception_weights_path",
                        str(as_path(weights.get("inception_weights_path"), root)),
                        "--num_frames",
                        str(facesim_cfg.get("num_frames", 16)),
                    ],
                    cwd=root / "evaluation/codes/vision/facesim",
                    face_csv=face_csv,
                    log_dir=log_dir,
                    parallel_jobs=facesim_jobs,
                    devices=device_list(facesim_cfg, facesim_device),
                )
                values = {
                    "facesim_cur": stats.get("cur_mean"),
                    "facesim_arc": stats.get("arc_mean"),
                    "facesim_fid": stats.get("fid_mean"),
                    "facesim_count": stats.get("count"),
                }
                row.update(values)
                cache_metric_result(detail_dir, metric_cache, "facesim", values)
        except subprocess.CalledProcessError as exc:
            row["facesim_error"] = short_error(exc.output or str(exc))
        except Exception as exc:
            row["facesim_error"] = short_error(str(exc))
        finally:
            metric_bar.update(1)

    if metric_enabled("wer", True):
        metric_bar.set_description(f"{name}: wer")
        try:
            wer_json = detail_dir / "wer_results.json"
            if not use_cached_metric("wer", row, detail_dir, metric_cache, skip_completed, len(matched_sample_ids)):
                wer_gt_dir = prepare_wer_gt(asr_dir, dataset_workspace / "wer_gt", matched_sample_ids)
                run_cmd_tee(
                    [
                        *metric_python_cmd(cfg, "wer", python_bin),
                        "wer.py",
                        "--mode",
                        "both",
                        "--gt_dir",
                        str(wer_gt_dir),
                        "--audio_dir",
                        str(audio_dir),
                        "--model_path",
                        str(weights.get("whisper_model_path", "large-v3")),
                        "--lang",
                        metrics["wer"].get("lang", "en"),
                        "--output",
                        str(detail_dir),
                        "--output_file",
                        wer_json.name,
                    ],
                    cwd=root / "evaluation/codes/audio/wer",
                    log_path=log_dir / "wer.log",
                )
                corrected = detail_dir / "wer_results_corrected.json"
                values = parse_wer_json(corrected if corrected.exists() else wer_json)
                row.update(values)
                cache_metric_result(detail_dir, metric_cache, "wer", values)
        except subprocess.CalledProcessError as exc:
            row["wer_error"] = short_error(exc.output or str(exc))
        except Exception as exc:
            row["wer_error"] = short_error(str(exc))
        finally:
            metric_bar.update(1)

    if metric_enabled("audio_sim", False):
        metric_bar.set_description(f"{name}: audio_sim")
        try:
            sim_txt = detail_dir / "audio_sim.txt"
            if not use_cached_metric("audio_sim", row, detail_dir, metric_cache, skip_completed, len(matched_sample_ids)):
                run_cmd_tee(
                    [
                        *metric_python_cmd(cfg, "audio_sim", python_bin),
                        "bench_verification.py",
                        "batch_verification",
                        "--model_name",
                        metrics["audio_sim"].get("model_name", "wavlm_large"),
                        "--audio1_dir",
                        str(audio_ref_eval_dir),
                        "--audio2_dir",
                        str(audio_dir),
                        "--checkpoint",
                        str(as_path(weights.get("speaker_verification_checkpoint"), root)),
                        "--use_gpu",
                        str(metrics["audio_sim"].get("use_gpu", True)),
                        "--output_file",
                        str(sim_txt),
                    ],
                    cwd=root / "evaluation/codes/audio/wer",
                    log_path=log_dir / "audio_sim.log",
                )
                values = parse_audio_sim(sim_txt)
                row.update(values)
                cache_metric_result(detail_dir, metric_cache, "audio_sim", values)
        except subprocess.CalledProcessError as exc:
            row["audio_sim_error"] = short_error(exc.output or str(exc))
        except Exception as exc:
            row["audio_sim_error"] = short_error(str(exc))
        finally:
            metric_bar.update(1)

    if metric_enabled("fad", False):
        metric_bar.set_description(f"{name}: fad")
        try:
            if not use_cached_metric("fad", row, detail_dir, metric_cache, skip_completed, len(matched_sample_ids)):
                fad_cfg = metrics["fad"]
                fad_json = detail_dir / "fad.json"
                cmd = [
                    *metric_python_cmd(cfg, "fad", python_bin),
                    "eval_fad_cli.py",
                    "--background_dir",
                    str(audio_ref_eval_dir),
                    "--eval_dir",
                    str(audio_dir),
                    "--output_json",
                    str(fad_json),
                    "--model_name",
                    fad_cfg.get("model_name", "vggish"),
                    "--sample_rate",
                    str(fad_cfg.get("sample_rate", 16000)),
                    "--dtype",
                    fad_cfg.get("dtype", "float32"),
                ]
                if fad_cfg.get("use_pca", False):
                    cmd.append("--use_pca")
                if fad_cfg.get("use_activation", False):
                    cmd.append("--use_activation")
                if fad_cfg.get("verbose", False):
                    cmd.append("--verbose")
                if fad_cfg.get("offline", False):
                    cmd.append("--offline")
                if weights.get("fad_hf_cache"):
                    cmd.extend(["--hf_cache", str(as_path(weights["fad_hf_cache"], root))])
                run_cmd_tee(
                    cmd,
                    cwd=root / "evaluation/codes/audio/fad",
                    log_path=log_dir / "fad.log",
                )
                values = json.loads(fad_json.read_text(encoding="utf-8"))
                values.setdefault("fad_count", len(matched_sample_ids))
                row.update(values)
                cache_metric_result(detail_dir, metric_cache, "fad", values)
        except Exception as exc:
            row["fad_error"] = short_error(str(exc))
        finally:
            metric_bar.update(1)

    if metric_enabled("fvd", False):
        metric_bar.set_description(f"{name}: fvd")
        try:
            if not use_cached_metric("fvd", row, detail_dir, metric_cache, skip_completed, len(matched_sample_ids)):
                real_video_dir = as_path(metrics["fvd"].get("real_video_dir"), root)
                if real_video_dir is None or not real_video_dir.exists():
                    raise RuntimeError("FVD is enabled but metrics.fvd.real_video_dir is empty or does not exist.")
                real_video_eval_dir = prepare_video_subset_workspace(
                    real_video_dir,
                    sample_ids,
                    matched_sample_ids,
                    dataset_workspace / "fvd_real_videos",
                )
                fvd_json = detail_dir / "fvd.json"
                fvd_cfg = metrics["fvd"]
                frame_size = fvd_cfg.get("frame_size", [224, 224])
                if fvd_cfg.get("method", "videogpt") == "styleganv":
                    i3d_weights_path = as_path(weights.get("fvd_styleganv_i3d_path"), root)
                else:
                    i3d_weights_path = as_path(weights.get("fvd_videogpt_i3d_path"), root)
                run_cmd_tee(
                    [
                        *metric_python_cmd(cfg, "fvd", python_bin),
                        "eval_fvd_cli.py",
                        "--real_dir",
                        str(real_video_eval_dir),
                        "--gen_dir",
                        str(flat_video_dir),
                        "--output_json",
                        str(fvd_json),
                        "--device",
                        fvd_cfg.get("device", cfg.get("device", "cuda")),
                        "--method",
                        fvd_cfg.get("method", "videogpt"),
                        "--only_final",
                        str(fvd_cfg.get("only_final", True)),
                        "--batch_size",
                        str(fvd_cfg.get("batch_size", 1)),
                        "--max_frames",
                        str(fvd_cfg.get("max_frames", 30)),
                        "--frame_size",
                        f"{frame_size[0]},{frame_size[1]}",
                        "--i3d_weights_path",
                        str(i3d_weights_path) if i3d_weights_path is not None else "",
                    ],
                    cwd=root / "evaluation/codes/vision/fvd",
                    log_path=log_dir / "fvd.log",
                )
                if not fvd_json.exists():
                    raise RuntimeError(f"FVD command finished but did not create {fvd_json}. Check {log_dir / 'fvd.log'}.")
                metric_values = json.loads(fvd_json.read_text(encoding="utf-8"))
                if "fvd" not in metric_values:
                    raise RuntimeError(f"FVD output is missing the 'fvd' field: {fvd_json}")
                row.update(metric_values)
                cache_metric_result(detail_dir, metric_cache, "fvd", metric_values)
        except Exception as exc:
            row["fvd_error"] = short_error(str(exc))
            print(f"FVD error for {name}: {row['fvd_error']}")
        finally:
            metric_bar.update(1)

    if metric_enabled("syncnet", False):
        metric_bar.set_description(f"{name}: syncnet")
        try:
            sync_json = detail_dir / "syncnet.json"
            if not use_cached_metric("syncnet", row, detail_dir, metric_cache, skip_completed, len(matched_sample_ids)):
                sync_cfg = metrics["syncnet"]
                syncnet_model_path = as_path(weights.get("syncnet_model_path"), root)
                if syncnet_model_path is None or not syncnet_model_path.exists():
                    raise RuntimeError("SyncNet is enabled but weights.syncnet_model_path is empty or does not exist.")
                sync_cmd = [
                    *metric_python_cmd(cfg, "syncnet", python_bin),
                    "batch_syncnet.py",
                    "--video_dir",
                    str(flat_video_dir),
                    "--output_file",
                    str(sync_json),
                    "--initial_model",
                    str(syncnet_model_path),
                    "--tmp_base_dir",
                    str(dataset_workspace / "syncnet_tmp"),
                    "--batch_size",
                    str(sync_cfg.get("batch_size", 10)),
                    "--vshift",
                    str(sync_cfg.get("vshift", 15)),
                    "--facedet_scale",
                    str(sync_cfg.get("facedet_scale", 0.25)),
                    "--crop_scale",
                    str(sync_cfg.get("crop_scale", 0.40)),
                    "--min_track",
                    str(sync_cfg.get("min_track", 100)),
                    "--frame_rate",
                    str(sync_cfg.get("frame_rate", 25)),
                    "--num_failed_det",
                    str(sync_cfg.get("num_failed_det", 25)),
                    "--min_face_size",
                    str(sync_cfg.get("min_face_size", 100)),
                ]
                if not sync_cfg.get("use_crop_pipeline", True):
                    sync_cmd.append("--no_crop_pipeline")
                if sync_cfg.get("allow_full_frame_fallback", False):
                    sync_cmd.append("--allow_full_frame_fallback")
                run_cmd_tee(
                    sync_cmd,
                    cwd=root / "evaluation/codes/syncnet",
                    log_path=log_dir / "syncnet.log",
                )
                values = parse_syncnet_json(sync_json)
                row.update(values)
                cache_metric_result(detail_dir, metric_cache, "syncnet", values)
        except subprocess.CalledProcessError as exc:
            row["syncnet_error"] = short_error(exc.output or str(exc))
        except Exception as exc:
            row["syncnet_error"] = short_error(str(exc))
        finally:
            metric_bar.update(1)

    metric_bar.close()

    return row


def rebuild_summary_from_existing(cfg: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    paths = cfg["paths"]
    metrics = cfg.get("metrics", {})
    output_root = as_path(paths["output_dir"], root)
    testdata_dir = as_path(paths["testdata_dir"], root)
    assert output_root is not None and testdata_dir is not None

    prompt_dir = testdata_dir / paths.get("prompt_dir_name", "full_video_prompt")
    sample_ids = sample_ids_from_prompt_dir(prompt_dir)
    metric_names = [
        (metric_name, default_enabled)
        for metric_name, default_enabled in [
            ("clip", True),
            ("facesim", True),
            ("wer", True),
            ("audio_sim", False),
            ("fad", False),
            ("fvd", False),
            ("syncnet", False),
        ]
        if metrics.get(metric_name, {}).get("enabled", default_enabled)
    ]

    rows = []
    for item in cfg.get("result_sets", []):
        if not item.get("enabled", True):
            continue
        name = item["name"]
        result_dir = as_path(item["path"], root)
        row: dict[str, Any] = {"name": name, "result_dir": str(result_dir), "error": ""}
        if result_dir is None or not result_dir.exists():
            row["error"] = "result_dir does not exist"
            rows.append(row)
            continue
        row.update(video_match_meta(result_dir, sample_ids))
        detail_dir = output_root / "details" / name
        metric_cache = load_metric_cache(detail_dir)
        for metric_name, _ in metric_names:
            values = cached_metric_result(metric_name, detail_dir, metric_cache)
            if values:
                row.update(values)
                cache_metric_result(detail_dir, metric_cache, metric_name, values)
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run configured evaluation metrics and summarize them as tables.")
    parser.add_argument("--config", default="evaluation/configs/evaluation.yaml")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override paths.output_dir from the config. A fresh directory avoids reusing old metric cache.",
    )
    parser.add_argument(
        "--workspace-dir",
        default=None,
        help="Override paths.workspace_dir. Defaults to <output-dir>/workspace when --output-dir is set.",
    )
    parser.add_argument(
        "--no-skip-completed",
        action="store_true",
        help="Ignore existing metric cache/results in the selected output directory and recompute enabled metrics.",
    )
    parser.add_argument(
        "--only-metric",
        action="append",
        choices=["clip", "facesim", "wer", "audio_sim", "fad", "fvd", "syncnet"],
        help="Run only this metric. Can be provided multiple times.",
    )
    parser.add_argument(
        "--rebuild-summary",
        action="store_true",
        help="Only rebuild summary.csv and summary.md from existing metric outputs/cache.",
    )
    args = parser.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    root = repo_root()
    cfg = load_config(as_path(args.config, root))
    apply_runtime_overrides(
        cfg,
        root,
        output_dir=args.output_dir,
        workspace_dir=args.workspace_dir,
        no_skip_completed=args.no_skip_completed,
    )
    only_metrics = set(args.only_metric or [])
    output_dir = as_path(cfg["paths"]["output_dir"], root)
    workspace_dir = as_path(cfg["paths"].get("workspace_dir", str(output_dir / "workspace")), root)
    assert output_dir is not None
    print(f"Evaluation output_dir: {output_dir}")
    print(f"Evaluation workspace_dir: {workspace_dir}")

    if args.rebuild_summary:
        rows = rebuild_summary_from_existing(cfg, root)
        write_table(rows, output_dir / "summary.csv", output_dir / "summary.md")
        print(f"\nSummary rebuilt from existing results: {output_dir / 'summary.csv'}")
        print(f"Markdown table rebuilt from existing results: {output_dir / 'summary.md'}")
        return

    rows = []
    enabled_items = [item for item in cfg.get("result_sets", []) if item.get("enabled", True)]
    for item in progress_iter(enabled_items, desc="Result sets", unit="set"):
        name = item["name"]
        result_dir = as_path(item["path"], root)
        if result_dir is None or not result_dir.exists():
            rows.append({"name": name, "result_dir": str(result_dir), "error": "result_dir does not exist"})
            continue
        print(f"\n===== Evaluating {name}: {result_dir} =====")
        rows.append(evaluate_one(name, result_dir, cfg, root, args.python_bin, only_metrics=only_metrics))

    write_table(rows, output_dir / "summary.csv", output_dir / "summary.md")
    print(f"\nSummary saved to {output_dir / 'summary.csv'}")
    print(f"Markdown table saved to {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
