from mpi4py import MPI
from . import Communicator, Status, Request
from functools import singledispatch

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


Request = MPI.Request


def _wrap_status(s: MPI.Status) -> Status:
    return _copy_status(Status(), s)

def _copy_status(s1: Status, s2: MPI.Status) -> Status:
    s1.count = s2.Get_count()
    s1.cancelled = s2.Is_cancelled()
    s1.COMMI_SOURCE = s2.Get_source()
    s1.COMMI_TAG = s2.Get_tag()
    s1.error = s2.Get_error()
    return s1
