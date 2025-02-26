import commi
from charm4py import charm
import numpy as np
import random

def ep(comm, id):
    rank = comm.Get_rank()
    size = comm.Get_size()

    # we want to cut the range [0,99] into size chunks
    # and then send each chunk to a different processor
    # so that each processor has a different chunk
    # and then we want to gather all the chunks back together
    # and make sure that each processor has the correct chunk
    data = np.arange(100)
    chunk_size = 100 // size
    chunks = [data[i*chunk_size:(i+1)*chunk_size] for i in range(size)]

    chunk = comm.scatter(chunks, root=0)
    assert np.all(chunk == np.arange(rank*chunk_size, (rank+1)*chunk_size))


    comm.barrier()
    recvd = comm.scatter(chunks, root=0)
    assert np.all(recvd == chunk)
    comm.barrier()


    data = list(range(0,100))
    chunk_size = 100 // size
    chunks = [data[i*chunk_size:(i+1)*chunk_size] for i in range(size)]

    chunk = comm.scatter(chunks, root=0)
    assert np.all(chunk == np.arange(rank*chunk_size, (rank+1)*chunk_size))


    comm.barrier()
    recvd = comm.scatter(chunks, root=0)
    assert np.all(recvd == chunk)
    comm.barrier()

    charm.exit()


def main(args):

    comm = commi.CreateCharmCommunicator([32], 32)
    comm.begin_exec(ep, 4)

commi.Start(main)
