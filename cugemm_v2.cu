//
// Created by sianh on 2023/3/6.
//

#include "cuda_runtime.h"
#include "iostream"
#include "vector"
#include "iomanip"

// in this sgemm BLKY==BLKX
template<int BLK>
__global__ void cu_sgemm(const float *a, const float *b, float *c, int M, int N, int K) {
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int by = blockIdx.y;
    int bx = blockIdx.x;

    __shared__ float shared_a[BLK][BLK];
    __shared__ float shared_b[BLK][BLK];

    const float *ptr_a = a + by * BLK * K;
    const float *ptr_b = b + bx * BLK;
    float sum = 0.f;

    for (int kk = 0; kk < K; kk += BLK) {
        shared_a[ty][tx] = ptr_a[ty * K + tx];
        shared_b[ty][tx] = ptr_b[ty * K + tx];
        __syncthreads();

#pragma unroll
        for (int i = 0; i < BLK; ++i) {
            sum += shared_a[ty][i] * shared_b[i][tx];
        }
        __syncthreads();

        ptr_a += BLK;
        ptr_b += BLK * N;
    }

    c[(BLK * by + ty) * N + BLK * bx + tx] = sum;
}

template<int BLKY, int BLKX>
class TestGemm {
public:
    TestGemm(int M, int N, int K) : M(M), N(N), K(K) {
        block.x = BLKX;
        block.y = BLKY;
        grid.x = (M + BLKX - 1) / BLKX;
        grid.y = (N + BLKY - 1) / BLKY;
    }

    void WarmUp();
    float Run(int n_iter);
    void InitHost();
    void InitDevice();
    void Free();

    int M, N, K;
    dim3 grid, block;
    float *d_A, *d_B, *d_C;
    float *h_A, *h_B, *h_C;
    cudaError_t err_;
};

template<int BLKY, int BLKX>
void MultiSizeTest(std::vector<float> &time) {
    time.clear();

    {
        TestGemm<BLKY, BLKX> test(256, 256, 256);
        test.InitHost();
        test.InitDevice();
        test.WarmUp();
        time.push_back(test.Run(10));
        test.Free();
    }

    {
        TestGemm<BLKY, BLKX> test(512, 512, 512);
        test.InitHost();
        test.InitDevice();
        test.WarmUp();
        time.push_back(test.Run(10));
        test.Free();
    }

    {
        TestGemm<BLKY, BLKX> test(1024, 1024, 1024);
        test.InitHost();
        test.InitDevice();
        test.WarmUp();
        time.push_back(test.Run(10));
        test.Free();
    }

    {
        TestGemm<BLKY, BLKX> test(2048, 2048, 2048);
        test.InitHost();
        test.InitDevice();
        test.WarmUp();
        time.push_back(test.Run(10));
        test.Free();
    }

    {
        TestGemm<BLKY, BLKX> test(4096, 4096, 4096);
        test.InitHost();
        test.InitDevice();
        test.WarmUp();
        time.push_back(test.Run(10));
        test.Free();
    }
    std::cout << std::endl << BLKY << "x" << BLKX << " " << "test results(ms)" << std::endl;
    std::cout << "256     512     1024     2048     4096" << std::endl;
    std::cout << "---------------------------------------" << std::endl;
    for (int i = 0; i < time.size(); ++i) {
        std::cout << std::setw(10) << std::setprecision(2) << time[i];
    }
    std::cout << std::endl;
}

int main() {
    {
        std::vector<float> time;
        MultiSizeTest<4, 4>(time);
    }
    {
        std::vector<float> time;
        MultiSizeTest<8, 8>(time);
    }
    {
        std::vector<float> time;
        MultiSizeTest<16, 16>(time);
    }
    {
        std::vector<float> time;
        MultiSizeTest<32, 32>(time);
    }

}

template<int BLKY, int BLKX>
void TestGemm<BLKY, BLKX>::InitHost() {
    h_A = new float[M * K];
    h_B = new float[K * N];
    h_C = new float[M * N];

    for (auto i = 0; i < M * K; ++i) {
        h_A[i] = 1.0f;
    }

    for (auto i = 0; i < K * N; ++i) {
        h_B[i] = 1.0f;
    }

    for (auto i = 0; i < M * N; ++i) {
        h_C[i] = 0.0f;
    }
}

template<int BLKY, int BLKX>
void TestGemm<BLKY, BLKX>::InitDevice() {
    if ((err_ = cudaMalloc((void **) &d_A, M * K * sizeof(float))) != cudaSuccess) {
        std::cout << "Failed to allocate device memory: "
                  << cudaGetErrorString(err_) << std::endl;
        exit(EXIT_FAILURE);
    }

    if ((err_ = cudaMalloc((void **) &d_B, K * N * sizeof(float))) != cudaSuccess) {
        std::cout << "Failed to allocate device memory: "
                  << cudaGetErrorString(err_) << std::endl;
        exit(EXIT_FAILURE);
    }

    if ((err_ = cudaMalloc((void **) &d_C, M * N * sizeof(float))) != cudaSuccess) {
        std::cout << "Failed to allocate device memory: "
                  << cudaGetErrorString(err_) << std::endl;
        exit(EXIT_FAILURE);
    }

    if ((err_ = cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) {
        std::cout << "Failed to copy dato to device memory: "
                  << cudaGetErrorString(err_) << std::endl;
        exit(EXIT_FAILURE);
    }

    if ((err_ = cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) {
        std::cout << "Failed to copy dato to device memory: "
                  << cudaGetErrorString(err_) << std::endl;
        exit(EXIT_FAILURE);
    }

    if ((err_ = cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) {
        std::cout << "Failed to copy dato to device memory: "
                  << cudaGetErrorString(err_) << std::endl;
        exit(EXIT_FAILURE);
    }
}

template<int BLKY, int BLKX>
void TestGemm<BLKY, BLKX>::Free() {
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

template<int BLKY, int BLKX>
void TestGemm<BLKY, BLKX>::WarmUp() {

//    std::cout << "matrix size: M = " << M << ", N = " << N << ", K = " << K << std::endl;
//    std::cout << "grid size: Z = " << grid.z << ", Y = " << grid.y << ", X = " << grid.x << std::endl;
//    std::cout << "block size: Z = " << block.z << ", Y = " << block.y << ", X = " << block.x << std::endl;
//    std::cout << " ----------- warmup() ----------- " << std::endl;
    cu_sgemm<BLKY><<<grid, block>>>(d_A, d_B, d_C, M, N, K);

    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < M * N; ++i) {
        if (fabs(h_C[i] - 1.0f * K) > 1e-6) {
            std::cout << "error in sgemm: " << BLKX << "x" << BLKY << std::endl;
        }
    }
}
template<int BLKY, int BLKX>
float TestGemm<BLKY, BLKX>::Run(int n_iter) {
    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

//    std::cout << " ----------- run test() ----------- " << std::endl;

    // 记录开始时刻的时间戳
    cudaEventRecord(start, 0);
    // Do Something

    for (int i = 0; i < n_iter; ++i) {
        cu_sgemm<BLKY><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }

    // 记录结束时刻的时间戳
    cudaEventRecord(stop, 0);
    // 等待事件同步值
    cudaEventSynchronize(stop);

    // 根据开始和结束时刻的时间戳，计算其中所经过的时间
    cudaEventElapsedTime(&elapsedTime, start, stop);
    // 打印时间
//    printf("Average run time: %6.2f ms\n", elapsedTime / float(n_iter));

    return elapsedTime / float(n_iter);
}