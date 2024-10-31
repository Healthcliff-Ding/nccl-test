#include <cuda_runtime.h>
#include <nccl.h>
#include <mpi.h>

#include <stdint.h>
#include <cassert>
#include <iostream>
#include <chrono>

constexpr size_t buf_size = 2147483648; // 2GB
static void* d_buf;
constexpr size_t g_buf_size = 16777216; // 16MB
static void* d_g_buf;
static ncclComm_t comm;
static cudaStream_t stream;

template<int RANK>
static void benchmark_inner(size_t chunk_size) {
    auto start = std::chrono::steady_clock::now();
    if (RANK == 0) {
        for (size_t i = 0; i < buf_size / chunk_size; ) {
            for (size_t j = 0; j < g_buf_size / chunk_size; j++, i++) {
                cudaMemcpyAsync((char*)d_g_buf + j * chunk_size,
                                (char*)d_buf + i * chunk_size,
                                chunk_size, cudaMemcpyDeviceToDevice);
            }
            ncclSend(d_g_buf, g_buf_size, ncclChar, 1, comm, stream);
        }
    } else {
        for (size_t i = 0; i < buf_size / chunk_size; ) {
            ncclRecv(d_g_buf, g_buf_size, ncclChar, 0, comm, stream);
            for (size_t j = 0; j < g_buf_size / chunk_size; j++, i++) {
                cudaMemcpyAsync((char*)d_buf + i * chunk_size,
                                (char*)d_g_buf + j * chunk_size,
                                chunk_size, cudaMemcpyDeviceToDevice);
            }
        }
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
    cudaMalloc(&d_g_buf, g_buf_size); // 16MB

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
    MPI_Barrier(MPI_COMM_WORLD);

    std::cout << "Benchmarking..." << std::endl;
    benchmark(rank, 128 * 1024);
    benchmark(rank, 256 * 1024);
    benchmark(rank, 512 * 1024);
    benchmark(rank, 1024 * 1024);
    benchmark(rank, 4 * 1024 * 1024);

    MPI_Finalize();
}