#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <random>
#include <functional>

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


int get_index(int x, int y, int z, int xdim, int ydim, int zdim)
{
    return (x * ydim * zdim) + (y * zdim) + z;
}

// Put the values of the neighbors of the given position into the given array
// It is the responsibility of the caller to set initial values of the neighbors,
// which is relevant for the edge cases which are missing some neighbors.
// Each neighbor has a specific location in the neighbors array (see the
// "n_idx = get_index(...)" line), in case some update functions want to abandon
// symmetry
int get_neighbors(bool *neighbors, int x, int y, int z, int xdim, int ydim, int zdim, bool *field)
{
    int i_start = x == 0 ? 0 : -1;
    int j_start = y == 0 ? 0 : -1;
    int k_start = z == 0 ? 0 : -1;
    int i_end = x == xdim - 1 ? 0 : 1;
    int j_end = y == ydim - 1 ? 0 : 1;
    int k_end = z == zdim - 1 ? 0 : 1;

    // This might make the GPU really sad...
    for (int i = i_start; i <= i_end; i++)
    {
        for (int j = j_start; j <= j_end; j++)
        {
            for (int k = k_start; k <= k_end; k++)
            {
                if (i == 0 && y == 0 && k == 0)
                    continue;
                // Find the index within the neighbors array 
                int n_idx = get_index(i + 1, j + 1, k + 1, 3, 3, 3);
                // And account for not including the middle element
                if (n_idx >= 13)
                    n_idx--;
                neighbors[n_idx] = field[get_index(x + i, y + j, z + k, xdim, ydim, zdim)];
            }
        }
    }
}

// TODO Have more interesting field initialization
void init_field(bool *field, int xdim, int ydim, int zdim)
{
    for (int i = 0; i < xdim; i++)
    {
        for (int j = 0; j < ydim; j++)
        {
            for (int k = 0; k < zdim; k++)
            {
                field[get_index(i, j, k, xdim, ydim, zdim)] = distribution(engine) < .3;
            }
        }
    }
}

int main(int argc, char **argv)
{
    int xdim, ydim, zdim, automaton_idx;
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

        if (argc != 5)
        {
            fprintf(stderr, "Usage: \"%s <x_dim> <y_dim> <z_dim> <automaton number>\" or \"%s list\"\n",
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
    }

#if DEBUG
    printf("Got xdim=%d, ydim=%d, zdim=%d, automaton_idx=%d\n", xdim, ydim, zdim, automaton_idx);
#endif

    bool **fields = (bool **) malloc(2 * sizeof(bool *));
    // Let's malloc to better match what the GPU code will look like.
    // Also the field can get pretty big. Wouldn't want to blow the stack.
    for (int i = 0; i < 2; i++)
    {
        // Malloc one contiguous block since that's what the GPU version will want
        fields[i] = (bool *) malloc(xdim * ydim * zdim * sizeof(bool));
    }

    // Initialize the rng and use it to populate the field
    engine.seed(0);
    
    init_field(fields[0], xdim, ydim, zdim);

    bool *old_field, *new_field;
    for (int count = 0; true; count++)
    {
#if DEBUG
        printf("Running generation %d\n", count);
#endif
        old_field = fields[count % 2];
        new_field = fields[(count + 1) % 2];
        for (int i = 0; i < xdim; i++)
        {
            for (int j = 0; j < ydim; j++)
            {
                for (int k = 0; k < zdim; k++)
                {
                    int idx = get_index(i, j, k, xdim, ydim, zdim);
                    bool neighbors[26] = {};
                    get_neighbors(neighbors, i, j, k, xdim, ydim, zdim, old_field);
                    new_field[get_index(i, j, k, xdim, ydim, zdim)] = aut.update_fun(neighbors);
                }
            }
        }
    }
}
