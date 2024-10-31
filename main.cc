#include <cuda_runtime.h>
#include <nccl.h>
#include <mpi.h>

#include <stdint.h>
#include <cassert>
#include <iostream>
#include <chrono>

constexpr size_t buf_size = 2147483648; // 2GB
static void* d_buf;
static ncclComm_t comm;
static cudaStream_t stream;

template<int RANK>
static void benchmark_inner(size_t chunk_size) {
    auto start = std::chrono::steady_clock::now();
    if (RANK == 0) {
        for (size_t i = 0; i < buf_size / chunk_size; i++)
            ncclSend(d_buf, chunk_size, ncclChar, 1, comm, stream);
    } else {
        for (size_t i = 0; i < buf_size / chunk_size; i++)
            ncclRecv(d_buf, chunk_size, ncclChar, 0, comm, stream);
    }
    cudaStreamSynchronize(stream);
    auto stop = std::chrono::steady_clock::now();
    std::cout << "Bandwidth w/ chunk size = " << int(chunk_size / 1024.0) << " kB: "
              << ((buf_size >> 30) * 1000.0 * 1000.0 / std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count())
              << " GB/s" << std::endl;
}

void benchmark(int rank, size_t chunk_size) {
    if (rank == 0) {
        benchmark_inner<0>(chunk_size);
    } else {
        benchmark_inner<1>(chunk_size);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char* argv[])
{
    int stat = MPI_Init(&argc, &argv);

    // Get the rank and size
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    assert((world_size == 2));

    // CUDA Initialize
    cudaSetDevice(rank);
    cudaMalloc(&d_buf, buf_size); // 2GB

    ncclUniqueId uid;
    if (rank == 0) {
        ncclGetUniqueId(&uid);
    }
    MPI_Bcast(&uid, sizeof(uid), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclCommInitRank(&comm, world_size, uid, rank);
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    std::cout << "Warmup...";
    if (rank == 0) {
        ncclSend(d_buf, buf_size, ncclChar, 1, comm, stream);
    } else {
        ncclRecv(d_buf, buf_size, ncclChar, 0, comm, stream);
    }
    std::cout << "\tdone!" << std::endl;

    std::cout << "Benchmarking..." << std::endl;
    benchmark(rank, 128 * 1024);
    benchmark(rank, 1024 * 1024);
    benchmark(rank, 4 * 1024 * 1024);
    benchmark(rank, 16 * 1024 * 1024);
    benchmark(rank, 64 * 1024 * 1024);

    MPI_Finalize();
}