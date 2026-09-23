from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval_fvd import calculate_fvd_from_dirs


def parse_frame_size(value: str) -> tuple[int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--frame_size must look like 224,224")
    return int(parts[0]), int(parts[1])


def str2bool(value: str) -> bool:
    value = str(value).lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute FVD for two video directories.")
    parser.add_argument("--real_dir", required=True)
    parser.add_argument("--gen_dir", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--method", default="videogpt", choices=["videogpt", "styleganv"])
    parser.add_argument("--only_final", type=str2bool, default=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=30)
    parser.add_argument("--frame_size", type=parse_frame_size, default=(224, 224))
    parser.add_argument("--i3d_weights_path", default="")
    args = parser.parse_args()

    result = calculate_fvd_from_dirs(
        real_dir=args.real_dir,
        gen_dir=args.gen_dir,
        device=args.device,
        method=args.method,
        only_final=args.only_final,
        batch_size=args.batch_size,
        max_frames=args.max_frames,
        frame_size=args.frame_size,
        i3d_weights_path=args.i3d_weights_path or None,
    )
    values = [float(value) for value in result["value"]]
    output = {
        "fvd": values[-1],
        "fvd_mean": sum(values) / len(values),
        "fvd_count": result.get("num_videos"),
        "fvd_real_videos": result.get("real_videos"),
        "fvd_gen_videos": result.get("gen_videos"),
        "fvd_matched_videos": result.get("matched_videos"),
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"FVD: {output['fvd']}")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
