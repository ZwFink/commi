from mpi4py import MPI
from . import Communicator, Status, Request
from functools import singledispatch
from typing import Any


class MPICommunicator:
    def __init__(self, mpi_comm):
        self._comm = mpi_comm

        self.send = self._comm.send
        self.Send = self._comm.Send
        self.Isend = self._comm.Isend
        self.Irecv = self._comm.Irecv
        self.isend = self._comm.isend
        self.irecv = self._comm.irecv
        self.Get_rank = self._comm.Get_rank
        self.Get_size = self._comm.Get_size
        self.iprobe = self._comm.iprobe

    def Recv(self, buf, source: int = 0, tag: int = 0, status: Status = None):
        mpi_status = None
        if status:
            mpi_status = MPI.Status()

        retv = self._comm.Recv(buf, source, tag, mpi_status)

        if status:
            _copy_status(status, mpi_status)
        return retv

    def recv(self, buf: Any = None, source: int = -1, tag: int = -1, status: Status = None):
        mpi_status = None
        if status:
            mpi_status = MPI.Status()

        retv = self._comm.recv(buf, source, tag, mpi_status)

        if status:
            _copy_status(status, mpi_status)
        return retv

    def get_communicator(self):
        return self._comm

    def Dup(self):
        return MPICommunicator(self._comm.Dup())

    def Free(self):
        pass


    def _copy_status(self, s1: Status, s2: MPI.Status) -> Status:
        s1.count = s2.Get_count()
        s1.cancelled = s2.Is_cancelled()
        s1.COMMI_SOURCE = s2.Get_source()
        s1.COMMI_TAG = s2.Get_tag()
        s1.error = s2.Get_error()
        return s1

Communicator.register(MPICommunicator)

Request = MPI.Request


def _wrap_status(s: MPI.Status) -> Status:
    return _copy_status(Status(), s)


def _copy_status(s1: Status, s2: MPI.Status) -> Status:
    s1.count = s2.Get_count()
    s1.cancelled = s2.Is_cancelled()
    s1.source = s2.Get_source()
    s1.COMMI_TAG = s2.Get_tag()
    s1.error = s2.Get_error()
    return s1
