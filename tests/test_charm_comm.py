import commi
from charm4py import charm
import numpy as np
import random

def ep(comm, id):
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        comm.Send(4, dest=1)
    if rank == 1:
        print("Rank 2 receiving!")
        a = comm.Recv(source=-1)
        print("received",a)

    print(comm.allgather(comm.Get_rank()))

    print(f"Hello from processor {rank} of {size} {id}")


    if rank == 0:
        bufs = [np.array([i]) for i in range(100)]
        for i in range(100):
            comm.Isend(bufs[i], dest=1, tag=i)
    if rank == 1:
        bufs = [np.array([0]) for i in range(100)]
        reqs = list()

        for i in range(100):
            reqs.append(comm.Irecv(bufs[i], source=0, tag=i))
        random.shuffle(reqs)

        for r in reqs:
            r.Wait()
            i = r.tag
            if bufs[i] != i:
                print("darnet")
        print("success")
    comm.barrier()
    if rank == 0:
        bufs = [np.array([2*i]) for i in range(100)]
        for i in range(100):
            comm.Isend(bufs[i], dest=1, tag=i)

    elif rank == 1:
        bufs = [np.array([0]) for i in range(100)]
        reqs = list()
        for i in range(100):
            reqs.append(comm.Irecv(bufs[i], source=0, tag=i))

        num_complete = 0
        while num_complete != 100:
            completed_idxes = commi.request.Waitsome(reqs)
            if completed_idxes:
                num_complete += len(completed_idxes)

                print(f"{len(completed_idxes)} requests were ready right away., total {num_complete} now done")
                print(completed_idxes)
                for i in completed_idxes:
                    if not reqs[i].Test():
                        print(f"Request i is not complete?")
                    if bufs[i] != 2*i:
                        print(f"{bufs[i]}, {i}:darnet")
    print("Done")
    comm.barrier()
    charm.exit(0)




def main(args):

    comm = commi.CreateCharmCommunicator([2], 2)
    comm.begin_exec(ep, 4)

commi.Start(main)
