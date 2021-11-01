from mpi4py import MPI
from . import Communicator

class MPICommunicator(Communicator):
    def __init__(self, mpi_comm):
        self._comm = mpi_comm

        self.send = self._comm.send
        self.recv = self._comm.recv
        self.Send = self._comm.Send
        self.Recv = self._comm.Recv
        self.Isend = self._comm.Isend
        self.Irecv = self._comm.Irecv
        self.isend = self._comm.isend
        self.irecv = self._comm.irecv
        self.Get_rank = self._comm.Get_rank

    def get_communicator(self):
        return self._comm


MPICommunicatorRequest = MPI.Request
