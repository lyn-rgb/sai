"""多卡 / NCCL 环境自检：十几秒内回答「这台机器现在的 8 卡通信是否正常」。

    python tests/check_nccl.py                 # 用全部可见 GPU
    GPUS=0,1 python tests/check_nccl.py        # 只用部分卡

它只做「每卡建 CUDA context → 在 NCCL 上 all_reduce → barrier」，不加载任何模型。
当训练在 `accelerator.prepare` / DeepSpeed 初始化阶段报下面这类错时，先跑这个脚本：

    ncclUnhandledCudaError: Call to CUDA function failed.
    Cuda failure 401 'the operation cannot be performed in the present state'

- 这个脚本也失败 → 节点/NCCL 层面的问题（GPU 被别的进程占着、Xid 报错、P2P/IB 配置），
  与仓库代码无关；
- 这个脚本通过、训练仍然失败 → 再回到代码/配置层面排查。

脚本同时打印每张卡的**空闲显存**：如果某张卡明显少于其他卡，说明上面还跑着别的进程，
这正是上面那类报错最常见的原因。
"""
from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

PORT = os.environ.get("NCCL_CHECK_PORT", "29511")


def worker(rank: int, world_size: int, device_ids: list[int]) -> None:
    device = device_ids[rank]
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{PORT}", rank=rank, world_size=world_size)

    tensor = torch.ones(8, device=device) * (rank + 1)
    dist.all_reduce(tensor)
    dist.barrier()

    free, total = torch.cuda.mem_get_info(device)
    expected = sum(range(1, world_size + 1))
    status = "OK" if float(tensor[0]) == expected else f"数值不符 ({float(tensor[0])} != {expected})"
    print(f"  [rank {rank}] cuda:{device} {torch.cuda.get_device_name(device)} "
          f"| all_reduce {status} | 空闲显存 {free / 2**30:.1f}/{total / 2**30:.1f} GiB", flush=True)
    dist.destroy_process_group()


def main() -> int:
    gpus_env = os.environ.get("GPUS")
    device_ids = ([int(x) for x in gpus_env.split(",")] if gpus_env else list(range(torch.cuda.device_count())))
    if not device_ids:
        print("❌ 没有可见的 GPU")
        return 1

    print("=== 环境 ===")
    print(f"  torch {torch.__version__} | CUDA {torch.version.cuda} | NCCL {'.'.join(map(str, torch.cuda.nccl.version()))}")
    print(f"  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(未设置)')}")
    print(f"  参与自检的卡: {device_ids}")
    for device in device_ids:
        free, total = torch.cuda.mem_get_info(device)
        print(f"    cuda:{device} 空闲 {free / 2**30:.1f}/{total / 2**30:.1f} GiB"
              + ("   ⚠️ 明显被占用" if free < total * 0.7 else ""))

    print(f"\n=== {len(device_ids)} 卡 NCCL all_reduce ===")
    mp.spawn(worker, args=(len(device_ids), device_ids), nprocs=len(device_ids), join=True)
    print("✅ NCCL 通信正常：若训练仍在 accelerator.prepare 处报错，请把完整日志贴出来")
    return 0


if __name__ == "__main__":
    sys.exit(main())
