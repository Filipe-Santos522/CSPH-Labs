#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"

#define THREADS_PER_BLOCK 256

#define DEBUG
#ifdef DEBUG
#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true) {
if (code != cudaSuccess) {
fprintf(stderr, "CUDA Error: %s at %s:%d\n",
cudaGetErrorString(code), file, line);
if (abort) exit(code);
}
}
#else
#define cudaCheckError(ans) ans
#endif

// This is the CUDA "kernel" function that is run on the GPU.  You
// know this because it is marked as a __global__ function.
__global__ void
upsweep_kernel(int N, int* input, int* output, int two_d, int two_dplus1) {

    __shared__ int vector[768];
    int idv=3*threadIdx.x;
    int idt= (blockIdx.x * blockDim.x + threadIdx.x)*two_dplus1 + (two_d-1);

    vector[idv] = input[idt]; 
    vector[idv+1] = input[idt+two_dplus1-two_d]; 
    // compute overall thread index from position of thread in current
    // block, and given the block we are in (in this example only a 1D
    // calculation is needed so the code only looks at the .x terms of
    // blockDim and threadIdx.
    vector[idv+1] += vector[idv];
    __syncthreads();
    for(int i=3; i<=idv; i*=2){
        if((idv+3)%(i*2)==0){
            vector[idv+1] += vector[idv-i+1];
        }
        __syncthreads();
    }
    input[idt+two_dplus1-two_d]= vector[idv+1];
}


// This is the CUDA "kernel" function that is run on the GPU.  You
// know this because it is marked as a __global__ function.
__global__ void
downsweep_kernel(int N, int* input, int* output, int two_d, int two_dplus1,int totalThreads) {

    // compute overall thread index from position of thread in current
    // block, and given the block we are in (in this example only a 1D
    // calculation is needed so the code only looks at the .x terms of
    // blockDim and threadIdx.
    __shared__ int vector[768];
    int idv=2*threadIdx.x;
    int index= (blockIdx.x * blockDim.x + threadIdx.x)*two_dplus1 + (N/totalThreads-1);

    vector[idv] = output[index];
    vector[idv+1] = output[index+two_dplus1-two_d];
    __syncthreads();
    // this check is necessary to make the code work for values of N
    // that are not a multiple of the thread block size (blockDim.x)
    //for();
    int t = vector[idv];
    output[index] = vector[idv+1];
    output[index+two_dplus1-two_d] += t;
}


// This is the CUDA "kernel" function that is run on the GPU.  You
// know this because it is marked as a __global__ function.
__global__ void
upsweep_kernel_sub(int N, int* input, int* output) {

    // compute overall thread index from position of thread in current
    // block, and given the block we are in (in this example only a 1D
    // calculation is needed so the code only looks at the .x terms of
    // blockDim and threadIdx.
    int index = blockIdx.x * blockDim.x + threadIdx.x;


    // this check is necessary to make the code work for values of N
    // that are not a multiple of the thread block size (blockDim.x)
    if(index<N-1){
        if(input[index]==input[index+1]){
            output[index]=1;
        }
        else{
            output[index]=0;
        }
    }
}

__global__ void
merge_kernel(int N, int* marked, int* scanned, int* output) {

    // compute overall thread index from position of thread in current
    // block, and given the block we are in (in this example only a 1D
    // calculation is needed so the code only looks at the .x terms of
    // blockDim and threadIdx.
    int index = blockIdx.x * blockDim.x + threadIdx.x;


    // this check is necessary to make the code work for values of N
    // that are not a multiple of the thread block size (blockDim.x)
    if(index<N && marked[index]==1)
        output[scanned[index]] = index;
}

// helper function to round an integer up to the next power of 2
static inline int nextPow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

// exclusive_scan --
//
// Implementation of an exclusive scan on global memory array `input`,
// with results placed in global memory `result`.
//
// N is the logical size of the input and output arrays, however
// students can assume that both the start and result arrays we
// allocated with next power-of-two sizes as described by the comments
// in cudaScan().  This is helpful, since your parallel segmented scan
// will likely write to memory locations beyond N, but of course not
// greater than N rounded up to the next power of 2.
//
// Also, as per the comments in cudaScan(), you can implement an
// "in-place" scan, since the timing harness makes a copy of input and
// places it in result

void exclusive_scan(int* input, int N, int* result)
{

    int threadsPerBlock = 256;
    
    // STUDENTS TODO:
    //
    // Implement your exclusive scan implementation here.  Keep in
    // mind that although the arguments to this function are device
    // allocated arrays, this is a function that is running in a thread
    // on the CPU.  Your implementation will need to make multiple calls
    // to CUDA kernel functions (that you must write) to implement the
    // scan.
    // upsweep phase
    int Nhelp=N;
    N=nextPow2(N);
    int totalThreads=8000;
    
    int blocks=0;

    blocks = ((Nhelp/2 + threadsPerBlock - 1) / threadsPerBlock);
    int two_dplus1 = 2;
    totalThreads=Nhelp/two_dplus1;
    
    if(totalThreads<THREADS_PER_BLOCK){
            threadsPerBlock=nextPow2(totalThreads);
        }else{
            threadsPerBlock = THREADS_PER_BLOCK;
        }
    int prevthreadsPerBlock=threadsPerBlock*2;
    for (int two_d = 1; two_d < N/2; two_d*=prevthreadsPerBlock) { 
        int prevthreadsPerBlock=threadsPerBlock*2;
        blocks = ((totalThreads + threadsPerBlock - 1) / threadsPerBlock);
        upsweep_kernel<<<blocks, threadsPerBlock>>>(N, result,result, two_d, two_dplus1);
        cudaCheckError(cudaDeviceSynchronize());
        two_dplus1 *=prevthreadsPerBlock;
        totalThreads=N/two_dplus1;
        if(totalThreads<THREADS_PER_BLOCK){
            threadsPerBlock=nextPow2(totalThreads);
        }else{
            threadsPerBlock = THREADS_PER_BLOCK;
        }

    }
    
    int zero=0;
    cudaMemcpy(result+N-1,&zero, sizeof(int), cudaMemcpyHostToDevice);
    totalThreads=1;
    blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    // downsweep phase
    for (int two_d = N/2; two_d >= 1; two_d /= 2) {
        int two_dplus1 = 2*two_d;
        if(totalThreads<THREADS_PER_BLOCK){
            threadsPerBlock=totalThreads;
        }else{
            threadsPerBlock = THREADS_PER_BLOCK;
        }
        blocks = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;
        downsweep_kernel<<<blocks, threadsPerBlock>>>(N, result,result, two_d, two_dplus1,totalThreads*2);
        cudaCheckError(cudaDeviceSynchronize());
        totalThreads=totalThreads*2;
    }

}


//
// cudaScan --
//
// This function is a timing wrapper around the student's
// implementation of segmented scan - it copies the input to the GPU
// and times the invocation of the exclusive_scan() function
// above. Students should not modify it.
double cudaScan(int* inarray, int* end, int* resultarray)
{
    int* device_result;
    int* device_input;
    int N = end - inarray;  

    // This code rounds the arrays provided to exclusive_scan up
    // to a power of 2, but elements after the end of the original
    // input are left uninitialized and not checked for correctness.
    //
    // Student implementations of exclusive_scan may assume an array's
    // allocated length is a power of 2 for simplicity. This will
    // result in extra work on non-power-of-2 inputs, but it's worth
    // the simplicity of a power of two only solution.

    int rounded_length = nextPow2(end - inarray);
    
    cudaMalloc((void **)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void **)&device_input, sizeof(int) * rounded_length);

    // For convenience, both the input and output vectors on the
    // device are initialized to the input values. This means that
    // students are free to implement an in-place scan on the result
    // vector if desired.  If you do this, you will need to keep this
    // in mind when calling exclusive_scan from find_repeats.
    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_input, N, device_result);

    // Wait for completion
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
       
    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


// cudaScanThrust --
//
// Wrapper around the Thrust library's exclusive scan function
// As above in cudaScan(), this function copies the input to the GPU
// and times only the execution of the scan itself.
//
// Students are not expected to produce implementations that achieve
// performance that is competition to the Thrust version, but it is fun to try.
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);
    
    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
   
    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int), cudaMemcpyDeviceToHost);

    thrust::device_free(d_input);
    thrust::device_free(d_output);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


// find_repeats --
//
// Given an array of integers `device_input`, returns an array of all
// indices `i` for which `device_input[i] == device_input[i+1]`.
//
// Returns the total number of pairs found
int find_repeats(int* device_input, int length, int* device_output) {

    // STUDENTS TODO:
    //
    // Implement this function. You will probably want to
    // make use of one or more calls to exclusive_scan(), as well as
    // additional CUDA kernel launches.
    //    
    // Note: As in the scan code, the calling code ensures that
    // allocated arrays are a power of 2 in size, so you can use your
    // exclusive_scan function with them. However, your implementation
    // must ensure that the results of find_repeats are correct given
    // the actual array length.
    int rounded_length = nextPow2(length);
    int repeats=0;
    int *marked,*partial;
    const int threadsPerBlock = THREADS_PER_BLOCK;
    const int blocks = (rounded_length + threadsPerBlock - 1) / threadsPerBlock;

    cudaMalloc(&marked, rounded_length*sizeof(int));

    upsweep_kernel_sub<<<blocks, threadsPerBlock>>>(length, device_input,marked);
    cudaDeviceSynchronize();
    
    cudaMalloc(&partial, rounded_length*sizeof(int));
    cudaMemcpy(partial, marked, length*sizeof(int), cudaMemcpyDeviceToDevice);

    exclusive_scan(marked,rounded_length,partial);

    merge_kernel<<<blocks, threadsPerBlock>>>(length, marked,partial,device_output);
    cudaDeviceSynchronize();
    cudaMemcpy(&repeats, partial + length-1, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(marked);
    cudaFree(partial);
    return repeats; 
}


//
// cudaFindRepeats --
//
// Timing wrapper around find_repeats. You should not modify this function.
double cudaFindRepeats(int *input, int length, int *output, int *output_length) {

    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    double startTime = CycleTimer::currentSeconds();
    
    int result = find_repeats(device_input, length, device_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    // set output count and results array
    *output_length = result;
    cudaMemcpy(output, device_output, length * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    float duration = endTime - startTime; 
    return duration;
}



void printCudaInfo()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n"); 
}