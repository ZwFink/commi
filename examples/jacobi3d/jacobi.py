import numpy as np
import time
from numba import cuda
import time
from charm4py import charm
# don't init mpi4py
import mpi4py
mpi4py.rc.initialize = False
from mpi4py import MPI
import commi
import kernels
from argparse import ArgumentParser
from enum import Enum
from functools import reduce
import sys


class Defaults(Enum):
    GRID_WIDTH = 64
    GRID_HEIGHT = 64
    GRID_DEPTH = 64
    NUM_ITERS = 100
    WARMUP_ITERS = 10
    LB_PERIOD = 10

def calc_num_procs_per_dim(num_procs, grid_w, grid_h, grid_d):
    """Calculates the 'optimal' 3D decomposition of processes."""
    n_procs_xyz = [0, 0, 0]
    area = [0.0, 0.0, 0.0]
    area[0] = float(grid_w * grid_h)
    area[1] = float(grid_w * grid_d)
    area[2] = float(grid_h * grid_d)

    bestsurf = 2.0 * sum(area)
    ipx = 1
    while ipx <= num_procs:
        if num_procs % ipx == 0:
            nremain = num_procs // ipx
            ipy = 1
            while ipy <= nremain:
                if nremain % ipy == 0:
                    ipz = nremain // ipy
                    if ipx * ipy * ipz == num_procs:
                        surf = area[0] / ipy + area[1] / ipz + area[2] / ipx
                        surf_orig = area[0]/(ipx*ipy) + area[1]/(ipx*ipz) + area[2]/(ipy*ipz)

                        if surf_orig < bestsurf:
                            bestsurf = surf_orig
                            n_procs_xyz[0] = ipx
                            n_procs_xyz[1] = ipy
                            n_procs_xyz[2] = ipz
                ipy += 1
        ipx += 1

    if reduce(lambda x, y: x*y, n_procs_xyz) != num_procs:
         if comm.Get_rank() == 0:
             print(f"ERROR: Could not find a valid 3D decomposition for {num_procs} processes.")
             print(f"Tried: {n_procs_xyz}")
         comm.Abort(1)

    if n_procs_xyz[0] == 0:
        n_procs_xyz[0] = num_procs
        n_procs_xyz[1] = 1
        n_procs_xyz[2] = 1


    return n_procs_xyz


def main(comm):
    rank = comm.Get_rank()
    size = comm.Get_size()
    n_physical_procs = comm.Get_physical_size()
    initial_proc_num = comm.Get_proc_num()


    if rank == 0:
        parser = ArgumentParser(description="Jacobi3D implementation in MPI/CUDA (Host Staging)")
        parser.add_argument('-x', '--grid_width', type=int, default=Defaults.GRID_WIDTH.value)
        parser.add_argument('-y', '--grid_height', type=int, default=Defaults.GRID_HEIGHT.value)
        parser.add_argument('-z', '--grid_depth', type=int, default=Defaults.GRID_DEPTH.value)
        parser.add_argument('-i', '--iterations', type=int, default=Defaults.NUM_ITERS.value)
        parser.add_argument('-w', '--warmup_iterations', type=int, default=Defaults.WARMUP_ITERS.value)
        parser.add_argument('-l', '--lb_period', type=int, default=Defaults.LB_PERIOD.value,
                            help='Load balancing period')
        args = parser.parse_known_args(sys.argv[1::])[0]
        config = {
            'grid_width': args.grid_width,
            'grid_height': args.grid_height,
            'grid_depth': args.grid_depth,
            'n_iters': args.iterations,
            'warmup_iters': args.warmup_iterations,
            'num_procs': size,
            'lb_period': args.lb_period
        }
    else:
        config = None

    config = comm.bcast(config, root=0)

    grid_width = config['grid_width']
    grid_height = config['grid_height']
    grid_depth = config['grid_depth']
    n_iters = config['n_iters']
    warmup_iters = config['warmup_iters']
    num_procs = config['num_procs']
    lb_period = config['lb_period']

    n_procs_xyz = calc_num_procs_per_dim(num_procs, grid_width, grid_height, grid_depth)
    n_procs_x, n_procs_y, n_procs_z = n_procs_xyz

    if n_procs_x * n_procs_y * n_procs_z != num_procs:
        if rank == 0:
            print(f"ERROR: Calculated proc grid dimensions {n_procs_xyz} do not match total processes {num_procs}")
        comm.Abort(1)

    if grid_width % n_procs_x != 0 or grid_height % n_procs_y != 0 or grid_depth % n_procs_z != 0:
         if rank == 0:
            print(f"ERROR: Grid dimensions ({grid_width}x{grid_height}x{grid_depth}) not evenly divisible by process grid ({n_procs_x}x{n_procs_y}x{n_procs_z})")
         comm.Abort(1)


    block_width = grid_width // n_procs_x
    block_height = grid_height // n_procs_y
    block_depth = grid_depth // n_procs_z

    x_surf_count = block_height * block_depth
    y_surf_count = block_width * block_depth
    z_surf_count = block_width * block_height

    ghost_counts = [x_surf_count] * 2 + [y_surf_count] * 2 + [z_surf_count] * 2
    ghost_sizes_bytes = [count * np.dtype(np.float64).itemsize for count in ghost_counts]


    if rank == 0:
        print("\n[MPI+CUDA 3D Jacobi example (Host Staging)]\n")
        print(f"Grid: {grid_width} x {grid_height} x {grid_depth}, "
              f"Processes: {n_procs_x} x {n_procs_y} x {n_procs_z} = {num_procs}, "
              f"Block: {block_width} x {block_height} x {block_depth}, "
              f"Iterations: {n_iters}, Warm-up: {warmup_iters}\n\n",
              flush=True)

    px = rank % n_procs_x
    py = (rank // n_procs_x) % n_procs_y
    pz = rank // (n_procs_x * n_procs_y)

    neighbors = [-1] * kernels.DIR_COUNT

    if px > 0: neighbors[kernels.LEFT] = rank - 1
    if px < n_procs_x - 1: neighbors[kernels.RIGHT] = rank + 1
    if py > 0: neighbors[kernels.TOP] = rank - n_procs_x
    if py < n_procs_y - 1: neighbors[kernels.BOTTOM] = rank + n_procs_x
    if pz > 0: neighbors[kernels.FRONT] = rank - (n_procs_x * n_procs_y)
    if pz < n_procs_z - 1: neighbors[kernels.BACK] = rank + (n_procs_x * n_procs_y)

    bounds = [False] * kernels.DIR_COUNT
    if px == 0: bounds[kernels.LEFT] = True
    if px == n_procs_x - 1: bounds[kernels.RIGHT] = True
    if py == 0: bounds[kernels.TOP] = True
    if py == n_procs_y - 1: bounds[kernels.BOTTOM] = True
    if pz == 0: bounds[kernels.FRONT] = True
    if pz == n_procs_z - 1: bounds[kernels.BACK] = True

    stream = cuda.default_stream()

    temp_size_elements = (block_width + 2) * (block_height + 2) * (block_depth + 2)
    d_temperature = cuda.device_array(temp_size_elements, dtype=np.float64)
    d_new_temperature = cuda.device_array(temp_size_elements, dtype=np.float64)

    h_ghosts = [cuda.pinned_array(ghost_counts[i], dtype=np.float64)
                for i in range(kernels.DIR_COUNT)]

    d_ghosts = [cuda.device_array(ghost_counts[i], dtype=np.float64)
                for i in range(kernels.DIR_COUNT)]


    kernels.invokeInitKernel(d_temperature, block_width, block_height, block_depth, stream)
    kernels.invokeInitKernel(d_new_temperature, block_width, block_height, block_depth, stream)
    kernels.invokeGhostInitKernels(d_ghosts, ghost_counts, stream)
    for i in range(kernels.DIR_COUNT):
        h_ghosts[i].fill(0)

    kernels.invokeBoundaryKernels(d_temperature, block_width, block_height, block_depth, bounds, stream)
    kernels.invokeBoundaryKernels(d_new_temperature, block_width, block_height, block_depth, bounds, stream)

    stream.synchronize()
    comm.Barrier()

    t_start_total = 0.0
    total_comm_time = 0.0


    for current_iter in range(n_iters + warmup_iters):
        if current_iter == warmup_iters:
            comm.Barrier()
            t_start_total = time.perf_counter()
            total_comm_time = 0.0

        comm_start_time = time.perf_counter()

        for i in range(kernels.DIR_COUNT):
            if not bounds[i]:
                kernels.invokePackingKernel(d_temperature, d_ghosts[i], i,
                                            block_width, block_height, block_depth, stream)
        stream.synchronize()

        for i in range(kernels.DIR_COUNT):
            if not bounds[i]:
                d_ghosts[i].copy_to_host(h_ghosts[i], stream=stream)
        stream.synchronize()

        active_send_reqs = []
        active_recv_reqs = []

        for direction in range(kernels.DIR_COUNT):
            neighbor_rank_send = neighbors[direction]
            if neighbor_rank_send != -1:
                send_buf = h_ghosts[direction]
                req = comm.Isend(send_buf, dest=neighbor_rank_send, tag=direction)
                active_send_reqs.append(req)

            opposite_direction = [kernels.RIGHT, kernels.LEFT, kernels.BOTTOM, kernels.TOP, kernels.BACK, kernels.FRONT][direction]
            neighbor_rank_recv = neighbors[opposite_direction]
            if neighbor_rank_recv != -1:
                recv_buf = h_ghosts[direction]
                req = comm.Irecv(recv_buf, source=neighbor_rank_recv, tag=direction)
                active_recv_reqs.append(req)

        if active_recv_reqs:
            commi.request.Waitall(active_recv_reqs)
        if active_send_reqs:
            commi.request.Waitall(active_send_reqs)

        comm_end_time = time.perf_counter()
        if current_iter >= warmup_iters:
            total_comm_time += (comm_end_time - comm_start_time)

        for direction in range(kernels.DIR_COUNT):
             opposite_direction = [kernels.RIGHT, kernels.LEFT, kernels.BOTTOM, kernels.TOP, kernels.BACK, kernels.FRONT][direction]
             neighbor_rank_recv = neighbors[opposite_direction]
             if neighbor_rank_recv != -1:
                 d_ghosts[direction].copy_to_device(h_ghosts[direction], stream=stream)
        stream.synchronize() 

        for direction in range(kernels.DIR_COUNT):
             opposite_direction = [kernels.RIGHT, kernels.LEFT, kernels.BOTTOM, kernels.TOP, kernels.BACK, kernels.FRONT][direction]
             neighbor_rank_recv = neighbors[opposite_direction]
             if neighbor_rank_recv != -1:
                kernels.invokeUnpackingKernel(d_temperature, d_ghosts[direction], direction,
                                              block_width, block_height, block_depth, stream)
        stream.synchronize() 


        d_temperature, d_new_temperature = d_new_temperature, d_temperature

        # Formula from Galvez 2018 Charmpy
        alpha_i = 1.0
        if num_procs > 0:
            proc_num_fraction = initial_proc_num / n_physical_procs
            if proc_num_fraction <= 0.2 or proc_num_fraction >= 0.8:
                alpha_i = 10.0
            else:
                effective_iter = max(0, current_iter - warmup_iters)
                alpha_i = max(1.0, 100.0 * proc_num_fraction + 5.0 * (effective_iter / 30.0))

        num_invocations = max(1, int(round(alpha_i)))
        if current_iter == 0:
            total_invocations = comm.redux(num_invocations, root=0, op=MPI.SUM)
            max_invocations = comm.redux(num_invocations, root=0, op=MPI.MAX)
            if rank == 0:
                average_invocations = max_invocations / (total_invocations / num_procs)
                print(f"Load imbalance factor: {average_invocations}")
        compute_start_time = time.perf_counter()

        for k in range(num_invocations):
            if k > 0:
                 d_temperature, d_new_temperature = d_new_temperature, d_temperature

            kernels.invokeJacobiKernel(d_temperature, d_new_temperature,
                                       block_width, block_height, block_depth, stream)


        stream.synchronize()
        compute_end_time = time.perf_counter()

        if current_iter % lb_period == 0:
            del d_temperature
            del d_new_temperature
            del h_ghosts
            del d_ghosts
            del stream

            comm.Migrate()
            stream = cuda.default_stream()
            d_temperature = cuda.device_array(temp_size_elements, dtype=np.float64)
            d_new_temperature = cuda.device_array(temp_size_elements, dtype=np.float64)
            h_ghosts = [cuda.pinned_array(ghost_counts[i], dtype=np.float64)
                        for i in range(kernels.DIR_COUNT)]
            d_ghosts = [cuda.device_array(ghost_counts[i], dtype=np.float64)
                for i in range(kernels.DIR_COUNT)]
            neighbor_rank_recv = None
            neighbor_rank_send = None
            recv_buf = None



    comm.Barrier() # Ensure all ranks finish loop
    t_end_total = time.perf_counter()

    elapsed_time = t_end_total - t_start_total
    avg_comm_time = total_comm_time / n_iters if n_iters > 0 else 0

    elapsed_time = t_end_total - t_start_total
    avg_comm_time = avg_comm_time / n_iters

    if rank == 0:
        avg_total_time_ms = (elapsed_time / n_iters) * 1e3 if n_iters > 0 else 0
        avg_comm_time_ms = avg_comm_time * 1e3

        print(f"Execution Summary (Max across ranks):")
        print(f"Total time (excluding warmup): {round(elapsed_time, 3)} s")
        if n_iters > 0:
            print(f"Average total time per iteration: {round(avg_total_time_ms, 3)} ms")
            print(f"Average communication time per iteration: {round(avg_comm_time_ms, 3)} ms")
            compute_time_ms = avg_total_time_ms - avg_comm_time_ms
            print(f"Average computation time per iteration: {round(compute_time_ms, 3)} ms")
        print("-" * 30)
    comm.Barrier()
    sys.exit(0)

def _main(args):
    import os
    num_chares = int(os.environ['NUM_CHARES'])
    print(f"Running simulation with {num_chares} chares")
    comm = commi.CreateCharmCommunicator([num_chares], num_chares)
    comm.begin_exec(main)

commi.Start(_main)