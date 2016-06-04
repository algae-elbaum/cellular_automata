# Cellular Automata

To compile: `make`

To list the available automata: `./cpu_automata list` (not available for gpu)

To run a CPU automaton: `./cpu_automata <x_dim> <y_dim> <z_dim> <automaton number>`

To run a GPU autamaton  `./gpu_automata <x_dim> <y_dim> <z_dim> <blocks> <threadsPerBlock>`

The CPU cellular automata will simulate a 3d cellular automaton of the given parameters,
but itt won't have any output. If you want any indication of what it is doing internally,
you'll have to add it yourself. This means there isn't even a guarantee that it works
properly, but that is ok. The CPU version's primary function is to have some base code
so that the program logic is all thought out before the GPU version.

The GPU cellular automata will do the same, given a cuda compatible GPU, but also will
display the automaton live to the screen. There wasn't enough time or positive utility
to wade through the bloat that is opengl to render this thing in 3d, and by the time
I'm finishing this up I don't think there's time to ask for ncurses to be installed.
Thus, the display is horribly silly: only layer 1 is displayed (layer 0 if xdim==1),
and it's printed with just a printf. Right now, the update function is such that
each layer runs it's own independent Conway's game of life, which is sort of proof that
this works as intended, it successfully looks like it's supposed to. There's an #if 1
at the top of gpu_automata_cuda.cu that can be changed to #if 0 to switch to some nonsense
other update function.

Performance:
Amusingly, the gpu version is about as fast as the cpu version, but only because the fps
is manually throttled in the gpu version's main loop. Without that throttle, the gpu is
definitely way faster than the cpu code, reaching generation 1000 near instantly,
while the cpu version only gets a few done per second. I didn't time anything to get
actual numbers though.
I didn't used shared memory since it would just overflow in most cases. See line 91 of
gpu_automata_cuda.cu for a more detailed explanation. If shared memory were large enough,
then that would have been the main optimization for this program. In reality, the main
optimization, if it can be called that, is that the field array is one big stretch of
memory rather than a multidimensional array, and that is to allow the cache friendly
behavior of always reading new_field elements in sequential cache-happy order. Finding
neighbors is less cache friendly, since one element is found as a neighbor of several
different elements, and by the time the update function has moved on to updating on a new
x, the neighbor that's being re-found will probably be out of the cache. An idea to deal
with this is only need to look at each element once, and associate with each element a
structure that records pointers to it's neighbors. Then if element A found B as a neighbor,
then at the same time B could have one of its neighbors set to A, and then when B's turn
comes to look for a neighbor then B won't have to look where A is. However, that would
take up massive memory and thereby end up with no better performance, since if I understand
right most of that would just land in the same memory that old_field is in anyways.
