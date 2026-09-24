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

各组合为什么值得试：
  NCCL_CUMEM_ENABLE=0  NCCL 2.21 起默认用 CUDA VMM 分配器，部分驱动/容器下 cuMemMap 会失败，
                       正是 401 'the operation cannot be performed in the present state'
  NCCL_P2P_DISABLE=1   关掉 GPU 间 P2P（NVLink/PCIe 直连异常时）
  NCCL_IB_DISABLE=1    关掉 IB/RDMA（单机 8 卡通常用不到）
  NCCL_SHM_DISABLE=1   关掉共享内存传输（容器里 /dev/shm 太小时有用）
"""
from __future__ import annotations

import os
import subprocess
import sys

import torch

ATTEMPTS = [
    ("默认", {}),
    ("NCCL_CUMEM_ENABLE=0", {"NCCL_CUMEM_ENABLE": "0"}),
    ("NCCL_P2P_DISABLE=1", {"NCCL_P2P_DISABLE": "1"}),
    ("NCCL_CUMEM_ENABLE=0 + NCCL_P2P_DISABLE=1",
     {"NCCL_CUMEM_ENABLE": "0", "NCCL_P2P_DISABLE": "1"}),
    ("NCCL_CUMEM_ENABLE=0 + NCCL_P2P_DISABLE=1 + NCCL_IB_DISABLE=1",
     {"NCCL_CUMEM_ENABLE": "0", "NCCL_P2P_DISABLE": "1", "NCCL_IB_DISABLE": "1"}),
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
        env = os.environ.copy()
        env.update(extra)
        env["GPUS"] = ",".join(map(str, device_ids))
        env["NCCL_DEBUG"] = env.get("NCCL_DEBUG", "WARN")
        try:
            proc = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--child", str(29511 + index)],
                env=env, capture_output=True, text=True, timeout=240)
        except subprocess.TimeoutExpired:
            print("  ❌ 超时（很可能是通信卡住，而不是配置错）")
            continue
        output = (proc.stdout or "") + (proc.stderr or "")
        for line in output.splitlines():
            if line.startswith("  [rank") or "Cuda failure" in line or "nccl" in line.lower()[:40]:
                print(line.rstrip(), flush=True)
        if proc.returncode == 0:
            print(f"\n✅ 用这组可以跑通：{name}")
            if extra:
                print("   把下面这行加进 run_multiperson_finetune.sh 的 export 区（多机分支里已有部分）：")
                print("   export " + " ".join(f"{k}={v}" for k, v in extra.items()))
            else:
                print("   默认配置就能跑通，训练报 401 的话请把完整日志贴出来（可能是启动后才出问题）")
            return 0
        print(f"  ❌ 失败（returncode={proc.returncode}）", flush=True)

    print("\n❌ 所有组合都失败：这是节点/NCCL 层面的问题，与仓库代码无关。")
    print("   建议：1) 换一个节点重试；2) 带上 `NCCL_DEBUG=INFO python tests/check_nccl.py` 的输出找集群管理员；")
    print("         3) 确认驱动版本与 CUDA 12.4 / NCCL 2.21.5 兼容（torch 2.6+cu124）")
    return 1


if __name__ == "__main__":
    sys.exit(main())
