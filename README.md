# Cellular Automata

To compile: `make`

To list the available automata: `./cpu_automata list` or `./gpu_automata list`

To run a CPU automaton: `./cpu_automata <x_dim> <y_dim> <z_dim> <automaton number>`

To run a GPU autamaton  `./gpu_automata <x_dim> <y_dim> <z_dim> <automaton number>`

(Maybe some sort of initial density parameter will be added to the run commands)

The CPU cellular automata will simulate a 3d cellular automaton of the given parameters,
but itt won't have any output. If you want any indication of what it is doing internally,
you'll have to add it yourself. This means there isn't even a guarantee that it works
properly, but that is ok. The CPU version's primary function is to have some base code
so that the program logic is all thought out before the GPU version.

The GPU cellular automata will do the same, given a cuda compatible GPU, but also will
display the automaton live to the screen. Optimally I will find a library to enable
a full 3d display of the automaton with mouse/WASD controls. Otherwise it'll display
one layer at a time and give the capability to change which layer is displayed.
