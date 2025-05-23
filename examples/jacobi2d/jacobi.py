import numpy
import time
from charm4py import charm
import commi
import sys
BLOCK_WIDTH = 0
BLOCK_HEIGHT = 0
DIVIDEBY5 = 0.2

def factor(r):
    fac1 = int(numpy.sqrt(r+1.0))
    fac2 = 0
    for fac1 in range(fac1, 0, -1):
        if r%fac1 == 0:
            fac2 = r/fac1
            break;
    return fac1, fac2

def to_1d(WIDTH, x, y):
    return WIDTH * y + x

def to_2d(WIDTH, i):
    return (i%WIDTH, i//WIDTH)

def main(comm):

    me = comm.Get_rank() #My ID
    np = comm.Get_size() #Number of processor, NOT numpy
    x, y = factor(np)

    x = int(x)
    y = int(y)

    # what are my coordinates in the processor grid?
    X, Y = to_2d(y, me)

    if me == 0:
        print(f"X, y: {x} {y}")

    if me==0:
        print('Python MPI/Numpy  Stencil execution on 2D grid')

        if len(sys.argv) < 3:
            print('argument count = ', len(sys.argv))
            sys.exit("Usage: ./stencil <# iterations> [<array dimension> or <array dimension X> <array dimension Y>]")
    iterations = int(sys.argv[1])
    if iterations < 1:
        sys.exit("ERROR: iterations must be >= 1")

    n = int(sys.argv[2])
    if len(sys.argv) > 3:
        m = int(sys.argv[3])
    else:
        m = n

    nsquare = n * m

    if nsquare < np:
        sys.exit("ERROR: grid size ", nsquare, " must be at least # ranks: ", Num_procs);
    if n % x:
        sys.exit(f"ERROR: grid size {n} does not evenly divide the number of chares in the x dimension {x}")
    if m % y:
        sys.exit(f"ERROR: grid size {m} does not evenly divide the number of chares in the y dimension {y}")

    warmup=10
    if me == 0:
        print('Number of processors        = ', np)
        print('Grid shape (x,y)            = ', x, y)
        print('Problem Domain (x,y)        = ', n, m)
        print('Number of warmup iterations = ', warmup)
        print('Number of iterations        = ', iterations)

    my_blocksize = (n//x)*(m//y)
    
    # Set block dimensions
    width = m//y
    height = n//x
    set_block_params(width, height)
    
    # Calculate the total size including ghost cells
    total_size = (BLOCK_HEIGHT+2) * (BLOCK_WIDTH+2)
    
    T = numpy.ones(total_size, dtype=numpy.float64)
    newT = numpy.ones(total_size, dtype=numpy.float64)

    enforce_BC(T)

    top_buf_out = numpy.zeros(width)
    top_buf_in = numpy.zeros(width)
    bot_buf_out = numpy.zeros(width)
    bot_buf_in = numpy.zeros(width)

    right_buf_out = numpy.zeros(height)
    right_buf_in = numpy.zeros(height)
    left_buf_out = numpy.zeros(height)
    left_buf_in = numpy.zeros(height)


    if Y < y-1:
        top_nbr   = to_1d(y, X, Y+1)
    if Y > 0:
        bot_nbr   = to_1d(y, X, Y-1)
    if X > 0:
        left_nbr  = to_1d(y, X-1, Y)
    if X < x-1:
        right_nbr = to_1d(y, X+1, Y)

    for i in range(warmup + iterations):
        if i == warmup:
            tst = time.time()

        send_reqs = list()
        recv_reqs = list()
        recv_status = commi.Status()
        if Y < y-1 :
            req0 = comm.Irecv(top_buf_in, source =top_nbr , tag =101)
            pack_top(T, top_buf_out)
            req1 = comm.Isend(top_buf_out, dest =top_nbr, tag =99)
            recv_reqs.append(req0)
            send_reqs.append(req1)

        if Y > 0 :
            req2 = comm.Irecv(bot_buf_in, source =bot_nbr , tag =99)
            pack_bottom(T, bot_buf_out)
            req3 = comm.Isend(bot_buf_out, dest =bot_nbr, tag =101)
            recv_reqs.append(req2)
            send_reqs.append(req3)

        if X < x-1 :
            req4 = comm.Irecv(right_buf_in, source =right_nbr , tag =1010)
            pack_right(T, right_buf_out)
            req5 = comm.Isend(right_buf_out, dest =right_nbr, tag =990)
            recv_reqs.append(req4)
            send_reqs.append(req5)

        if X > 0 :
            req6 = comm.Irecv(left_buf_in, source =left_nbr , tag =990)
            pack_left(T, left_buf_out)
            req7 = comm.Isend(left_buf_out, dest =left_nbr, tag =1010)
            recv_reqs.append(req6)
            send_reqs.append(req7)

        idxes = commi.request.Waitall(recv_reqs)
        for idx in idxes:
            tag = recv_reqs[idx].tag
            if tag == 101:
                unpack_top(T, top_buf_in)
            elif tag == 99:
                unpack_bottom(T, bot_buf_in)
            elif tag == 990:
                unpack_left(T, left_buf_in)
            elif tag == 1010:
                unpack_right(T, right_buf_in)


        commi.request.Waitall(send_reqs)
        compute(newT, T)

        newT, T = T, newT
        enforce_BC(T)

        if i and i % 10 == 0:
            comm.Migrate()
            # it's cheaper to re-allocate these things than to serialize them
            # by re-defining them here, they values become dead
            # so, they are excluded from serialization
            top_buf_out = numpy.zeros(width)
            top_buf_in = numpy.zeros(width)
            bot_buf_out = numpy.zeros(width)
            bot_buf_in = numpy.zeros(width)

            right_buf_out = numpy.zeros(height)
            right_buf_in = numpy.zeros(height)
            left_buf_out = numpy.zeros(height)
            left_buf_in = numpy.zeros(height)
            newT = numpy.ones(total_size, dtype=numpy.float64)

    tend = time.time()
    if me == 0:
        print(f"Elapsed: {tend-tst}")
    comm.barrier()
    charm.exit()



def set_block_params(width, height):
    global BLOCK_WIDTH
    global BLOCK_HEIGHT
    BLOCK_WIDTH = width
    BLOCK_HEIGHT = height


def index(x, y):
    # Convert 2D coordinates to 1D index, accounting for ghost cells
    return (x+1)*(BLOCK_WIDTH+2) + (y+1)


def pack_left(temperature, ghost):
    for x in range(BLOCK_HEIGHT):
        ghost[x] = temperature[index(x, 0)]


def pack_right(temperature, ghost):
    for x in range(BLOCK_HEIGHT):
        ghost[x] = temperature[index(x, BLOCK_WIDTH-1)]


def pack_top(temperature, ghost):
    for y in range(BLOCK_WIDTH):
        ghost[y] = temperature[index(BLOCK_HEIGHT-1, y)]


def pack_bottom(temperature, ghost):
    for y in range(BLOCK_WIDTH):
        ghost[y] = temperature[index(0, y)]


def unpack_left(temperature, ghost):
    for x in range(BLOCK_HEIGHT):
        temperature[index(x, -1)] = ghost[x]


def unpack_right(temperature, ghost):
    for x in range(BLOCK_HEIGHT):
        temperature[index(x, BLOCK_WIDTH)] = ghost[x]


def unpack_top(temperature, ghost):
    for y in range(BLOCK_WIDTH):
        temperature[index(BLOCK_HEIGHT, y)] = ghost[y]


def unpack_bottom(temperature, ghost):
    for y in range(BLOCK_WIDTH):
        temperature[index(-1, y)] = ghost[y]


def compute(new_temperature, temperature):
    for i in range(BLOCK_HEIGHT):
        for j in range(BLOCK_WIDTH):
            new_temperature[index(i, j)] = (temperature[index(i-1, j)] \
                                          + temperature[index(i+1, j)] \
                                          + temperature[index(i, j-1)] \
                                          + temperature[index(i, j+1)] \
                                          + temperature[index(i, j)]) \
                                          * DIVIDEBY5


def enforce_BC(temperature):
    # heat the left and top faces of the block
    for y in range(BLOCK_WIDTH):
        temperature[index(-1, y)] = 255.0
    for x in range(BLOCK_HEIGHT):
        temperature[index(x, -1)] = 255.0


def _main(args):
    comm = commi.CreateCharmCommunicator([16], 16)
    comm.begin_exec(main)

commi.Start(_main)
