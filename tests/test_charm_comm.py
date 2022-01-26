import commi
from charm4py import charm

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


def main(args):

    comm = commi.CreateCharmCommunicator([2], 2)
    comm.begin_exec(ep, 4)

commi.Start(main)
