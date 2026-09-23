"""Smoke tests for the multi-person reference path.

Run inside the training environment (it needs torch and the repo's usual imports):

    python tests/test_multiperson_refs.py
    python tests/test_multiperson_refs.py --gpu --ckpt-dir ./ckpts   # also the VAE guard

The tests cover the invariants the multi-person design relies on:

1. reference faces stacked on the frame dim land on *distinct* temporal RoPE positions, so no
   positional code has to change for N references;
2. the reference-pair fusion is block diagonal: person j never sees person k != j, and a change to
   person k's reference leaves person j's output untouched;
3. the fusion adapters only exist when the feature is enabled and are a no-op at initialization,
   so starting from a single-person checkpoint cannot perturb the trained behaviour (item 3 of the
   plan's verification ladder; the numerical comparison against the pre-change code still has to be
   done on the real checkpoint, this test only pins the "cheap" half of it).
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.fusion import FusionModel
from modules.model import rope_apply_3d


def tiny_configs(dim=64, num_heads=8, num_layers=2):
    def tower(model_type, patch_size, in_dim, extra=None):
        config = dict(model_type=model_type, patch_size=patch_size, text_len=64, in_dim=in_dim,
                      dim=dim, ffn_dim=dim * 2, freq_dim=64, text_dim=64, out_dim=in_dim,
                      num_heads=num_heads, num_layers=num_layers, window_size=(-1, -1),
                      qk_norm=True, cross_attn_norm=True, eps=1e-6)
        config.update(extra or {})
        return config

    return (tower("t2v", [1, 2, 2], 8),
            tower("t2a", [1], 6, {"temporal_rope_scaling_factor": 0.19676}))


def tiny_fusion(seed=0, use_ref_av_fusion=False, n_refs=1, device="cpu"):
    torch.manual_seed(seed)
    video_config, audio_config = tiny_configs()
    model = FusionModel(video_config=video_config, audio_config=audio_config)
    model.eval()
    model.n_refs = n_refs
    model.init_lora(self_lora=False, train=False, vid_ip_emb_dim=8, audio_ip_emb_dim=6,
                    use_ref_av_fusion=use_ref_av_fusion, fusion_lora_rank=4)
    return model.to(device)


def test_stacked_reference_faces_get_distinct_positions():
    """Reference face j must be RoPE'd at the main clip's end + j, not at a shared position."""
    model = tiny_fusion()
    freqs = model.video_model.freqs
    torch.manual_seed(0)
    x = torch.randn(1, 2, 8, 8)                     # 2 reference faces, 1 token each
    main_grid = torch.tensor([[3, 1, 1]])           # main clip has 3 latent frames

    stacked = rope_apply_3d(x, torch.tensor([[2, 1, 1]]), freqs, offsets=main_grid)
    ref0 = rope_apply_3d(x[:, :1], torch.tensor([[1, 1, 1]]), freqs, offsets=main_grid)
    ref1 = rope_apply_3d(x[:, 1:], torch.tensor([[1, 1, 1]]), freqs, offsets=torch.tensor([[4, 1, 1]]))

    assert torch.allclose(stacked[:, :1], ref0, atol=1e-6), "reference 0 is not at the main clip's end"
    assert torch.allclose(stacked[:, 1:], ref1, atol=1e-6), "reference 1 is not one frame after reference 0"
    assert not torch.allclose(ref0, ref1, atol=1e-6), "the two references got the same position"


def test_reference_pair_attention_is_block_diagonal():
    """Person j's reference video tokens must only attend to person j's reference audio tokens."""
    # `flash_attention` is a CUDA kernel, so this one needs a GPU
    device = "cuda" if torch.cuda.is_available() else None
    if device is None:
        print("      (skipped: no CUDA device, flash_attention is CUDA only)")
        return
    model = tiny_fusion(use_ref_av_fusion=True, n_refs=2, device=device)
    vid_cross = model.video_model.blocks[0].cross_attn
    vid_freqs = model.video_model.freqs.to(device)
    audio_freqs = model.audio_model.freqs.to(device)
    torch.manual_seed(0)
    vid_ip = torch.randn(1, 8, model.video_model.dim, device=device)        # 2 persons x 4 tokens
    audio_ip = torch.randn(1, 6, model.audio_model.dim, device=device)      # 2 persons x 3 tokens
    vid_grid = torch.tensor([[2, 2, 2]], device=device)
    audio_grid = torch.tensor([[6]], device=device)
    vid_offsets = torch.tensor([[3, 0, 0]], device=device)
    audio_offsets = torch.tensor([[5]], device=device)

    def run(audio_refs):
        with torch.no_grad():
            return model.ref_pair_cross_attention(
                vid_cross, vid_ip, vid_grid, vid_freqs, vid_offsets,
                audio_refs, audio_grid, audio_freqs, audio_offsets, n_refs=2)

    out_a = run(audio_ip)
    perturbed = audio_ip.clone()
    perturbed[:, 3:] += 5.0                                  # touch person 1's voice only
    out_b = run(perturbed)

    assert torch.allclose(out_a[:, :4], out_b[:, :4], atol=1e-5), \
        "person 0's reference reacted to person 1's reference (pairing leaks between persons)"
    assert not torch.allclose(out_a[:, 4:], out_b[:, 4:], atol=1e-5), \
        "person 1's reference ignored its own reference audio"


def test_fusion_adapters_are_optional_and_start_as_a_noop():
    """The adapters must not exist when the feature is off, and must be zero at init when it is on."""
    off = tiny_fusion(use_ref_av_fusion=False)
    for block in list(off.video_model.blocks) + list(off.audio_model.blocks):
        assert not hasattr(block.cross_attn, "k_fusion_lora"), \
            "fusion adapters were created even though use_ref_av_fusion is disabled"
        assert not hasattr(block.cross_attn, "v_fusion_lora"), \
            "fusion adapters were created even though use_ref_av_fusion is disabled"

    on = tiny_fusion(use_ref_av_fusion=True)
    torch.manual_seed(0)
    for tower, dim in ((on.video_model, on.video_model.dim), (on.audio_model, on.audio_model.dim)):
        seq = torch.randn(1, 5, dim)
        for block in tower.blocks:
            assert hasattr(block.cross_attn, "k_fusion_lora"), "fusion adapters are missing"
            assert torch.count_nonzero(block.cross_attn.k_fusion_lora(seq)) == 0, \
                "the fusion LoRA must be zero initialized so a resumed checkpoint is unchanged"
            assert torch.count_nonzero(block.cross_attn.v_fusion_lora(seq)) == 0, \
                "the fusion LoRA must be zero initialized so a resumed checkpoint is unchanged"


def test_video_vae_encodes_one_frame_per_call(ckpt_dir: Path):
    """Guard for `modules/vae2_2.py`: several reference faces in one `wrapped_encode` lose all but the first.

    This is the reason the multi-person path encodes each reference separately.
    """
    from utils.model_loading_utils import init_wan_vae_2_2

    vae = init_wan_vae_2_2(str(ckpt_dir), rank=torch.cuda.current_device())
    refs = torch.randn(1, 3, 2, 256, 256, device=torch.cuda.current_device()).to(torch.bfloat16)
    with torch.no_grad():
        joint = vae.wrapped_encode(refs)
        separate = torch.cat([vae.wrapped_encode(refs[:, :, i:i + 1]) for i in range(2)], dim=2)

    assert joint.shape[2] == 1, \
        f"the video VAE now keeps several frames in one call ({joint.shape}), revisit the per-reference encode"
    assert separate.shape[2] == 2, \
        f"encoding the references one by one did not give one latent frame each ({separate.shape})"


def main():
    parser = argparse.ArgumentParser(description="Smoke tests for the multi-person reference path.")
    parser.add_argument("--gpu", action="store_true", help="also run the checks that need a GPU and real checkpoints")
    parser.add_argument("--ckpt-dir", type=Path, default=Path("ckpts"))
    args = parser.parse_args()

    tests = [
        test_stacked_reference_faces_get_distinct_positions,
        test_reference_pair_attention_is_block_diagonal,
        test_fusion_adapters_are_optional_and_start_as_a_noop,
    ]
    if args.gpu:
        tests.append(lambda: test_video_vae_encodes_one_frame_per_call(args.ckpt_dir))

    failed = []
    for test in tests:
        name = getattr(test, "__name__", "test_video_vae_encodes_one_frame_per_call")
        try:
            test()
            print(f"PASS  {name}")
        except Exception:
            failed.append(name)
            print(f"FAIL  {name}")
            traceback.print_exc()

    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
