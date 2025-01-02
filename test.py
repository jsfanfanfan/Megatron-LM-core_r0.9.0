import torch
import torch.distributed as dist
import os

def setup(rank, world_size):
    """初始化分布式环境"""
    os.environ['MASTER_ADDR'] = 'localhost'  # 主节点地址
    os.environ['MASTER_PORT'] = '2234'     # 通信端口
    dist.init_process_group("nccl", rank=rank, world_size=world_size)  # 使用 NCCL 后端（推荐用于 GPU）

def cleanup():
    """清理分布式环境"""
    dist.destroy_process_group()

def all_reduce_example(rank, world_size):
    """在多 GPU 上执行 all-reduce 操作"""
    setup(rank, world_size)

    # 每个 GPU 创建一个张量
    device = torch.device(f"cuda:{rank}")
    tensor = torch.randn((10000, 10000), device=device) * (rank + 1)  # 每个 GPU 上的张量值不同

    print(f"Before all-reduce on rank {rank}: {tensor}")

    # 执行 all-reduce 操作（默认是求和）
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    print(f"After all-reduce on rank {rank}: {tensor}")

    cleanup()

if __name__ == "__main__":
    world_size = 4  # 假设有 4 个 GPU
    torch.multiprocessing.spawn(all_reduce_example, args=(world_size,), nprocs=world_size, join=True)
