from __future__ import annotations

import argparse
import math
import random
import wave
from pathlib import Path


def read_pcm_samples(raw: bytes, sample_width: int) -> tuple[list[int], int, int]:
    if sample_width not in {1, 2, 3, 4}:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    samples = []
    if sample_width == 1:
        for value in raw:
            samples.append(value - 128)
        return samples, -128, 127

    min_value = -(1 << (sample_width * 8 - 1))
    max_value = (1 << (sample_width * 8 - 1)) - 1
    for idx in range(0, len(raw), sample_width):
        chunk = raw[idx : idx + sample_width]
        samples.append(int.from_bytes(chunk, byteorder="little", signed=True))
    return samples, min_value, max_value


def write_pcm_samples(samples: list[int], sample_width: int) -> bytes:
    if sample_width == 1:
        return bytes(max(0, min(255, value + 128)) for value in samples)

    chunks = []
    for value in samples:
        chunks.append(int(value).to_bytes(sample_width, byteorder="little", signed=True))
    return b"".join(chunks)


def rms(samples: list[int]) -> float:
    if not samples:
        return 0.0
    return math.sqrt(sum(float(value) * float(value) for value in samples) / len(samples))


def add_noise_to_wav(
    input_path: Path,
    output_path: Path,
    snr_db: float,
    rng: random.Random,
) -> dict[str, float]:
    with wave.open(str(input_path), "rb") as reader:
        params = reader.getparams()
        raw = reader.readframes(reader.getnframes())

    samples, min_value, max_value = read_pcm_samples(raw, params.sampwidth)
    signal_rms = rms(samples)
    noise_rms = signal_rms / (10.0 ** (snr_db / 20.0)) if signal_rms > 0 else 0.0

    noisy_samples = []
    clipped = 0
    for sample in samples:
        noisy = int(round(sample + rng.gauss(0.0, noise_rms)))
        if noisy < min_value:
            noisy = min_value
            clipped += 1
        elif noisy > max_value:
            noisy = max_value
            clipped += 1
        noisy_samples.append(noisy)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as writer:
        writer.setparams(params)
        writer.writeframes(write_pcm_samples(noisy_samples, params.sampwidth))

    return {
        "signal_rms": signal_rms,
        "noise_rms": noise_rms,
        "clipped_samples": float(clipped),
        "total_samples": float(len(samples)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create noisy copies of evaluation/testdata audio WAV files.")
    parser.add_argument("--input-dir", type=Path, default=Path("evaluation/testdata/audio"))
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/testdata/audio_noise"))
    parser.add_argument("--snr-db", type=float, default=10.0, help="Target signal-to-noise ratio in dB.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input audio directory does not exist: {input_dir}")

    wav_paths = sorted(input_dir.glob("*_audio.wav"))
    if not wav_paths:
        raise RuntimeError(f"No *_audio.wav files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    written = 0
    skipped = 0
    total_clipped = 0.0
    total_samples = 0.0

    for wav_path in wav_paths:
        output_path = output_dir / wav_path.name
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue
        stats = add_noise_to_wav(wav_path, output_path, args.snr_db, rng)
        total_clipped += stats["clipped_samples"]
        total_samples += stats["total_samples"]
        written += 1

    clip_rate = total_clipped / total_samples if total_samples else 0.0
    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"SNR: {args.snr_db} dB")
    print(f"Written: {written}")
    print(f"Skipped existing: {skipped}")
    print(f"Clipped sample rate: {clip_rate:.6f}")


if __name__ == "__main__":
    main()
