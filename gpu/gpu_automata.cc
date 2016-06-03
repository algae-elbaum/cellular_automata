#include <curand.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <random>
#include <functional>

#include "gpu_automata_cuda.cuh"

#define DEBUG 1

using namespace std;

typedef struct
{
    string name;
    bool (*update_fun)(bool *neighbors);
} automaton;

uniform_real_distribution<float> distribution(0, 1);
mt19937 engine;


// TODO Make interesting update functions for automata
int num_automata = 1;
automaton automata[] = 
{
    {
        "kary",
        // C++ HAS LAMBDAS! For the first time in forever, there'll be magic there'll be fun!
        [](bool *neighbors) -> bool {return true;}
        // This is beautiful
        // How didn't I know
        // C++ never seemed reasonable before
        // Now it has a redeeming quality over C
        // Amazing
    }
};


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

void init_field(bool *field, int size, int blocks, int threadsPerBlock)
{
    curandGenerator_t generator;
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    float *rand_floats;
    cudaMalloc(&rand_floats, sizeof(float) * size);
    cuda_call_init_field_kernel(blocks, threadsPerBlock, rand_floats, field, size);
}

int main(int argc, char **argv)
{
    int xdim, ydim, zdim, automaton_idx, blocks, threadsPerBlock;
    automaton aut;

    // Process args
    {
        if (argc == 2 && strcmp(argv[1], "list") == 0)
        {
            // List available automata
            for (int i = 0; i < num_automata; i++)
                printf("%d: %s\n", i + 1, automata[i].name.c_str());
            exit(EXIT_SUCCESS);
        }

        if (argc != 7)
        {
            fprintf(stderr, "Usage: \"%s <y_dim> <x_dim> <z_dim> <automaton number> <blocks> <threadsPerBlock>\" or \"%s list\"\n",
                    argv[0], argv[0]);
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

        automaton_idx = atoi(argv[4]);
        if (automaton_idx <= 0 || automaton_idx > num_automata)
        {
            fprintf(stderr,"Invalid automaton number\n");
            exit(EXIT_FAILURE);
        }
        aut = automata[automaton_idx - 1];

        blocks = atoi(argv[5]);
        if (blocks <= 0)
        {
            fprintf(stderr,"Invalid number of blocks\n");
            exit(EXIT_FAILURE);
        }

        threadsPerBlock = atoi(argv[6]);
        if (threadsPerBlock <= 0)
        {
            fprintf(stderr,"Invalid number of threads per block\n");
            exit(EXIT_FAILURE);
        }


    }

#if DEBUG
    printf("Got xdim=%d, ydim=%d, zdim=%d, automaton_idx=%d, blocks=%d, threadsPerBlock=%d\n",
            xdim, ydim, zdim, automaton_idx, blocks, threadsPerBlock);
#endif

    bool *fields[2];
    for (int i = 0; i < 2; i++)
    {
        cudaMalloc(&fields[i], xdim * ydim * zdim * sizeof(bool));
    }

    init_field(fields[0], xdim * ydim * zdim, blocks, threadsPerBlock);

    bool *old_field, *new_field;
    for (int count = 0; true; count++)
    {
        old_field = fields[count % 2];
        new_field = fields[(count + 1) % 2];
        cuda_call_automaton_step_kernel(blocks, threadsPerBlock, old_field, new_field,
            xdim, ydim, zdim, aut.update_fun);
     }
}
