import commi
from charm4py import charm
import numpy as np
import random


def ep(comm, id):
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        comm.Send(4, dest=1)
        val = comm.Recv(source=1)
        assert val == 5
        print("success")
    if rank == 1:
        print("Rank 2 receiving!")
        a = comm.Recv(source=-1)
        print("received",a)
        comm.Send(5, dest=0, tag=250)

    if rank == 0:
        comm.Send(4, dest=1)
        st = commi.Status()

        val = comm.Recv(status=st)
        assert st.source == 1 and st.tag == -1 and val == 5
        print("success")
    if rank == 1:
        st = commi.Status()
        print("Rank 2 receiving!")
        a = comm.Recv(status = st)
        print("received",a, 'from', st.source, 'tag', st.tag)
        assert st.source == 0 and st.tag == -1 and a == 4
        comm.Send(5, dest=0)

    comm.barrier()
    charm.exit(0)



def main(args):

    comm = commi.CreateCharmCommunicator([2], 2)
    comm.begin_exec(ep, 4)

commi.Start(main)
