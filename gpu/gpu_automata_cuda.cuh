
__device__ bool blep_aut(bool *neighbors);

void cuda_call_init_field_kernel(const unsigned int blocks,
        const unsigned int threadsPerBlock, float *rands,
        bool *field, int length);

void cuda_call_automaton_step_kernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        bool *old_field, bool *new_field, int xdim, int ydim, int zdim);

