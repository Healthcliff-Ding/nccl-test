import torch
import torch.distributed as dist

import os
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '2'
os.environ['MASTER_PORT'] = '29501'
os.environ['MASTER_ADDR'] = 'localhost' # use net0
os.environ['NCCL_SOCKET_IFNAME'] = 'net0'
os.environ['NCCL_DEBUG'] = 'info'
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_SHM_DISABLE'] = '1'
os.environ['NCCL_IB_HCA'] = 'mlx5_2'

def pkg_chunk_test(_size: int):
    raw_data = torch.empty(2 * 1024 * 1024 * 1024, dtype=torch.int8).cuda()
    data = raw_data.view((-1, _size))
    start = time.time()
    for i in range(data.size(0)):
        dist.isend(data[i], dst=1)
    torch.cuda.synchronize()
    stop = time.time()
    print("chunk size: {}kB, bandwidth: {}GB/s".format(
        _size // 1024, 2 / (stop - start)
    ))

if __name__ == '__main__':
    options = dist.ProcessGroupNCCL.Options()
    options.is_high_priority_stream = True
    dist.init_process_group('nccl', pg_options=options)
    print('Rank1 waiting for barrier...')
    dist.barrier()
    torch.cuda.synchronize()

    print('Warmup...')
    pkg_chunk_test(2 << 30)
    
    print('Start...')
    pkg_chunk_test(128 * 1024)
    pkg_chunk_test(1024 * 1024)
