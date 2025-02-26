import mpi4py
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    comm.send(4, dest=1)
if rank == 1:
    print("Rank 2 receiving!")
    a = comm.recv(source=-1)
    print("received",a)

print(comm.allgather(comm.Get_rank()))

print(f"Hello from processor {rank} of {size} {id}")

buf1 = np.array([0])
buf2 = np.array([0])
if rank == 0:
    buf1[0] = 4
    buf2[0] = 5
    comm.Isend(buf1, dest=1, tag=4)
    comm.Isend(buf2, dest=1, tag=5)

    comm.Isend(buf1, dest=1, tag=5)
    comm.Isend(buf2, dest=1, tag=4)
if rank == 1:
    r1 = comm.Irecv(buf1, source=0, tag=5)
    r2 = comm.Irecv(buf2, source=0, tag=4)
    d2 = r2.Wait()
    d1 = r1.Wait()
    print(f"First I received {buf1}, and then {buf2}")

    r1 = comm.Irecv(buf1, source=0, tag=5)
    r2 = comm.Irecv(buf2, source=0, tag=4)
    d2 = r2.Wait()
    d1 = r1.Wait()
    print(f"First I received {buf1}, and then {buf2}")

