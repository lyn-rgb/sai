"""多卡 / NCCL 环境自检：先判断「节点通信是否正常」，再自动找出能跑通的环境变量组合。

    python tests/check_nccl.py                 # 用全部可见 GPU
    GPUS=0,1 python tests/check_nccl.py        # 只用部分卡

每个组合都做「每卡建 CUDA context → NCCL all_reduce → barrier」，不加载任何模型。
当训练在 `accelerator.prepare` / DeepSpeed 初始化阶段报下面这类错时先跑它：

    ncclUnhandledCudaError: Call to CUDA function failed.
    Cuda failure 401 'the operation cannot be performed in the present state'

- 本脚本**默认组合**也失败 → 节点/NCCL 层面的问题，与仓库代码无关（模型都还没开始建）；
  脚本会继续尝试 NCCL 2.21 上最常见的几个开关，并把**能跑通的那一组**打印出来，
  把它加进 `run_multiperson_finetune.sh` 的 export 区即可；
- 所有组合都失败 → 把输出贴给集群管理员（附 `NCCL_DEBUG=INFO` 的日志），或换个节点。

各组合为什么值得试（默认失败时，前几组会把 NCCL 的告警打出来，失败位置很有信息量：
出现在 `transport/nvls.cc` 就说明是 NVLS）：
  NCCL_NVLS_ENABLE=0   关掉 NVLS/NVLink SHARP（cuMulticast* 在部分驱动/容器下会返回 401，
                       代价最小，NVLS 只是优化）
  NCCL_CUMEM_ENABLE=0  NCCL 2.21 起默认用 CUDA VMM 分配器，部分驱动/容器下 cuMemMap 也会失败
  NCCL_P2P_DISABLE=1   关掉 GPU 间 P2P —— 会明显掉吞吐，尽量别用
  NCCL_IB_DISABLE=1    关掉 IB/RDMA（单机 8 卡通常用不到）
  NCCL_SHM_DISABLE=1   关掉共享内存传输（容器里 /dev/shm 太小时有用）—— 同样掉吞吐

命中的那组会**自动复查**（默认重复 2 次，`REPEAT=0` 可关），只有复查也通过才算稳定；
偶发失败会被标成 ⚠️，那种情况更可能是驱动/GPU 状态问题而不是配置问题。
"""
from __future__ import annotations

import os
import subprocess
import sys

import torch

ATTEMPTS = [
    ("默认", {}),
    # 从窄到宽：先试代价最小的（NVLS 只是优化，关掉几乎不影响单机吞吐；
    # 关 P2P/SHM 会让 GPU 间流量绕道主机内存，性能损失明显，放最后）
    ("NCCL_NVLS_ENABLE=0", {"NCCL_NVLS_ENABLE": "0"}),
    ("NCCL_NVLS_ENABLE=0 + NCCL_CUMEM_ENABLE=0",
     {"NCCL_NVLS_ENABLE": "0", "NCCL_CUMEM_ENABLE": "0"}),
    ("NCCL_NVLS_ENABLE=0 + NCCL_P2P_DISABLE=1",
     {"NCCL_NVLS_ENABLE": "0", "NCCL_P2P_DISABLE": "1"}),
    ("NCCL_CUMEM_ENABLE=0", {"NCCL_CUMEM_ENABLE": "0"}),
    ("NCCL_P2P_DISABLE=1", {"NCCL_P2P_DISABLE": "1"}),
    ("NCCL_CUMEM_ENABLE=0 + NCCL_P2P_DISABLE=1 + NCCL_IB_DISABLE=1 + NCCL_SHM_DISABLE=1",
     {"NCCL_CUMEM_ENABLE": "0", "NCCL_P2P_DISABLE": "1", "NCCL_IB_DISABLE": "1", "NCCL_SHM_DISABLE": "1"}),
]


def worker(rank: int, world_size: int, device_ids: list[int], port: str) -> None:
    from datetime import timedelta

    import torch.distributed as dist

    device = device_ids[rank]
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world_size,
                            timeout=timedelta(seconds=120))   # 卡住时快速失败，便于逐组排查

    tensor = torch.ones(8, device=device) * (rank + 1)
    dist.all_reduce(tensor)
    dist.barrier()

    expected = sum(range(1, world_size + 1))
    value = float(tensor[0])
    status = "OK" if value == expected else f"数值不符 ({value} != {expected})"
    print(f"  [rank {rank}] cuda:{device} | all_reduce {status}", flush=True)
    dist.destroy_process_group()


def child_main(port: str, device_ids: list[int]) -> int:
    import torch.multiprocessing as mp

    mp.spawn(worker, args=(len(device_ids), device_ids, port), nprocs=len(device_ids), join=True)
    return 0


def run_attempt(extra: dict, device_ids: list[int], port: int):
    """在独立子进程里用指定环境变量跑一次「每卡建 context + all_reduce」。"""
    env = os.environ.copy()
    env.update(extra)
    env["GPUS"] = ",".join(map(str, device_ids))
    env["NCCL_DEBUG"] = env.get("NCCL_DEBUG", "WARN")
    try:
        proc = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", str(port)],
                              env=env, capture_output=True, text=True, timeout=240)
    except subprocess.TimeoutExpired:
        return False, "  ❌ 超时（很可能是通信卡住，而不是配置错）"
    return proc.returncode == 0, (proc.stdout or "") + (proc.stderr or "")


def _print_highlights(output: str) -> None:
    """只挑关键行（每卡结果、NCCL 告警、失败位置）并去掉重复的姿态，避免刷屏。"""
    seen = set()
    for line in output.splitlines():
        if line.startswith("  [rank") or "Cuda failure" in line or "NCCL WARN" in line:
            # 8 个 rank 的 WARN 只有 pid/rank 不同，按告警本体去重
            key = line.split("NCCL WARN", 1)[1] if "NCCL WARN" in line else line
            if key in seen:
                continue
            seen.add(key)
            print(line.rstrip(), flush=True)
    if "nvls.cc" in output:
        print("   → 失败位置在 transport/nvls.cc（NVLS / NVLink SHARP），优先试 NCCL_NVLS_ENABLE=0", flush=True)


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        device_ids = [int(x) for x in os.environ["GPUS"].split(",")]
        return child_main(sys.argv[2], device_ids)

    gpus_env = os.environ.get("GPUS")
    device_ids = ([int(x) for x in gpus_env.split(",")] if gpus_env else list(range(torch.cuda.device_count())))
    if not device_ids:
        print("❌ 没有可见的 GPU")
        return 1

    print("=== 环境 ===")
    print(f"  torch {torch.__version__} | CUDA {torch.version.cuda} | "
          f"NCCL {'.'.join(map(str, torch.cuda.nccl.version()))}")
    print(f"  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(未设置)')}")
    print(f"  参与自检的卡: {device_ids}")
    for device in device_ids:
        free, total = torch.cuda.mem_get_info(device)
        print(f"    cuda:{device} 空闲 {free / 2**30:.1f}/{total / 2**30:.1f} GiB"
              + ("   ⚠️ 明显被别的进程占用" if free < total * 0.7 else ""))

    for index, (name, extra) in enumerate(ATTEMPTS):
        print(f"\n=== 第 {index + 1}/{len(ATTEMPTS)} 组：{name} ===", flush=True)
        ok, output = run_attempt(extra, device_ids, port=29511 + index)
        _print_highlights(output)
        if ok:
            repeats = max(0, int(os.environ.get("REPEAT", "2")))
            stable = True
            for repeat in range(repeats):
                ok_repeat, output = run_attempt(extra, device_ids, port=29611 + index * 10 + repeat)
                print(f"  复查 {repeat + 1}/{repeats}: {'通过' if ok_repeat else '失败'}", flush=True)
                if not ok_repeat:
                    _print_highlights(output)
                    stable = False
                    break
            print(f"\n✅ 用这组可以跑通：{name}" + ("（复查全部通过，可放心用于训练）" if stable else "（⚠️ 复查失败 → 可能是偶发，别只信一次）"))
            if extra:
                print("   直接跑训练：")
                print("   " + " ".join(f"{k}={v}" for k, v in extra.items()) + " STEPS=train bash run_multiperson_finetune.sh")
                print("   （要固化的话就加进 run_multiperson_finetune.sh 的 export 区）")
            else:
                print("   默认配置就能跑通，训练报 401 的话请把完整日志贴出来（可能是启动后才出问题）")
            return 0 if stable else 1
        print(f"  ❌ 失败（returncode!=0）", flush=True)

    print("\n❌ 所有组合都失败：这是节点/NCCL 层面的问题，与仓库代码无关。")
    print("   建议：1) 换一个节点重试；2) 带上 `NCCL_DEBUG=INFO python tests/check_nccl.py` 的输出找集群管理员；")
    print("         3) 确认驱动版本与 CUDA 12.4 / NCCL 2.21.5 兼容（torch 2.6+cu124）")
    return 1


if __name__ == "__main__":
    sys.exit(main())
