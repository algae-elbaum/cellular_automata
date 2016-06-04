#include <curand.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <random>
#include <functional>
#include <time.h>
#include <unistd.h>

#include "gpu_automata_cuda.cuh"

#define DEBUG 1
#define US_BW_GEN 300000

using namespace std;

/* Check errors on CUDA runtime functions */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

/* Check errors on CUDA kernel calls */
void checkCUDAKernelError()
{
    cudaError_t err = cudaGetLastError();
    if  (cudaSuccess != err){
        fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
    } else {
        fprintf(stderr, "No kernel error detected\n");
    }

}


void print_lvl(bool *display_lvl, int ydim, int zdim)
{
    char lvl_str[(ydim + 1) * zdim];
    int line = 0;
    for (int i = 0; i < ydim * zdim; i++)
    {
        lvl_str[i + line] = display_lvl[i] ? 'x' : ' ';
        if (i % zdim == 0)
        {
            line ++;
            lvl_str[i + line] = '\n';
        }
    }
    printf("%s\n----------------------------------\n", lvl_str);
}

void init_field(bool *field, int size, int blocks, int threadsPerBlock)
{
    curandGenerator_t generator;
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    float *rand_floats;
    cudaMalloc(&rand_floats, sizeof(float) * size);
    curandGenerateUniform(generator, rand_floats, size);
    cuda_call_init_field_kernel(blocks, threadsPerBlock, rand_floats, field, size);
}

int main(int argc, char **argv)
{
    int xdim, ydim, zdim, blocks, threadsPerBlock;

    // Process args
    {
        if (argc != 7)
        {
            fprintf(stderr, "Usage: \"%s <y_dim> <x_dim> <z_dim> <blocks> <threadsPerBlock>\n",
                    argv[0]);
            exit(EXIT_FAILURE);
        }

        // atoi returns zero on error, so these ifs do double duty in detecting errors
        xdim = atoi(argv[1]);
        if (xdim == 0)
        {
            fprintf(stderr,"Invalid x dimension\n");
            exit(EXIT_FAILURE);
        }

        ydim = atoi(argv[2]); 
        if (ydim <= 0)
        {
            fprintf(stderr,"Invalid y dimension\n");
            exit(EXIT_FAILURE);
        }

        zdim = atoi(argv[3]);
        if (zdim <= 0)
        {
            fprintf(stderr,"Invalid z dimension\n");
            exit(EXIT_FAILURE);
        }

        blocks = atoi(argv[4]);
        if (blocks <= 0)
        {
            fprintf(stderr,"Invalid number of blocks\n");
            exit(EXIT_FAILURE);
        }

        threadsPerBlock = atoi(argv[5]);
        if (threadsPerBlock <= 0)
        {
            fprintf(stderr,"Invalid number of threads per block\n");
            exit(EXIT_FAILURE);
        }
    }

#if DEBUG
    printf("Got xdim=%d, ydim=%d, zdim=%d, blocks=%d, threadsPerBlock=%d\n",
            xdim, ydim, zdim, blocks, threadsPerBlock);
#endif

    bool *fields[2];
    for (int i = 0; i < 2; i++)
    {
        gpuErrchk(cudaMalloc(&fields[i], xdim * ydim * zdim * sizeof(bool)));
    }

    init_field(fields[0], xdim * ydim * zdim, blocks, threadsPerBlock);
    checkCUDAKernelError();

    bool *old_field, *new_field;
    bool display_lvl[ydim * zdim];
    clock_t t1 = clock();
    for (int count = 0; count < 1040; count++)
    {
        // Aim for US_BW_GEN microseconds between generations. This basically
        // defeats the purpose of using the gpu, but whatever
        usleep(max(0, (int)(US_BW_GEN - (difftime(clock(), t1))/1000000)));
#if DEBUG
        printf("Running generation %d\n", count);
#endif
        old_field = fields[count % 2];
        new_field = fields[(count + 1) % 2];
        cuda_call_automaton_step_kernel(blocks, threadsPerBlock, old_field, new_field,
            xdim, ydim, zdim);
        checkCUDAKernelError();

        // I really want to be using ncurses (optimally opengl, but that's not
        // happening), but it's not installed and it's probably too late to ask
        // for it to be installed.
        // So instead I'm doing the awful awful thing of just printing it straight
        // to terminal. And part of that is that letting the user change level
        // is not doable in any nice way that I know of. Therefore I'm going to
        // make the executive decision that the level displayed will be the one
        // at x = 1 (or 0 if xdim = 1).
        int lvl = xdim == 1 ? 0 : 1;
        cudaMemcpy(display_lvl, new_field + (lvl * ydim * zdim),
                ydim * zdim * sizeof(bool), cudaMemcpyDeviceToHost);
        print_lvl(display_lvl, ydim, zdim);
     }
}
