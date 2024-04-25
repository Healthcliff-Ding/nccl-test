import torch
import torch.distributed as dist

import os
import time

os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '2'
os.environ['MASTER_PORT'] = '29501'
os.environ['MASTER_ADDR'] = '10.36.8.243' # use net0
os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
os.environ['NCCL_DEBUG'] = 'info'
os.environ['NCCL_IB_HCA'] = 'mlx5'

if __name__ == '__main__':
    options = dist.ProcessGroupNCCL.Options()
    options.is_high_priority_stream = True
    dist.init_process_group('nccl', pg_options=options)
    print('Rank0 waiting for barrier...')
    dist.barrier()
    torch.synchronize()
    for i in range(1, 10):
        tensor = torch.empty([i * 1024, 1024, 1024], dtype=torch.int8).cuda()
        start = time.time()
        dist.send(tensor, dst=1)
        torch.cuda.synchronize()
        stop = time.time()
        print("chunk size: {}GB, bandwidth: {}GB/s".format(
            i, i / (stop - start)
        ))
