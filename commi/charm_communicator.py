from charm4py import Chare, coro, Channel, charm, Reducer, Future
from . import Communicator
from . import Request, SendRequest, RecvRequest, RecvManager
from typing import Any
import numpy as np



class CharmCommunicator(Chare):
    def __init__(self, n_elems):
        self._mgr = RecvManager()
        self._comm = self.thisProxy
        self._size = n_elems
        self._this_index = self.thisIndex[0]

        self._channels_map = dict()
        self._channels = list()

        # TODO: CollectiveResult class
        self._allgather_fut = None
        self._allgather_result = None

        # TODO: How can we do wildcard recvs without this?
        for i in range(n_elems):
            if i != self.Get_rank():
                self._get_channel_to(i)

    def Send(self, buf: Any = None, dest: int = -1, tag: int = -1, status: Any = None):
        ch = self._get_channel_to(dest)
        ch.send(tag, buf)
    def Recv(self, buf: Any = None, source: int = -1, tag: int = -1, status: Any = None):
        if source == -1:
            gen = charm.iwait(self._channels)
            tag, recv = next(gen).recv()
            del gen
            return recv
        recv = self._mgf.receiveFromChannelWithTag(self._get_channel_to(source), tag)
        # ch = self._get_channel_to(source)
        # tag, recv = ch.recv()
        return recv


    def Irecv(self, buf: Any = None, source: int = -1, tag: int = -1, status: Any = None):
        return RecvRequest(buf, self._mgr, self._get_channel_to(source), tag)

    def Isend(self, buf: Any = None, dest: int = -1, tag: int = -1, status: Any = None):
        if tag == 0:
            print("Sending a with tag 0!")
        self.Send(buf, dest, tag, status)
        return SendRequest()

    def barrier(self):
        self.allreduce().get()


    def Get_rank(self):
        return self.thisIndex[0]
    def Get_size(self):
        return self._size

    def get_communicator(self):
        return self._comm

    def Dup(self):
        # NOTE: This is dangerous
        return self._comm

    def Free(self):
        pass

    @coro
    def allgather(self, sendobj: Any):
        # TODO: Expose LocalFuture to user code
        self._allgather_fut = Future()
        self.reduce(self.thisProxy._return_from_allgather, sendobj, Reducer.gather)
        self._allgather_fut.get()
        return self._allgather_result

    def _return_from_allgather(self, result):
        self._allgather_result = result
        self._allgather_fut()

    @coro
    def begin_exec(self, fn, *args, **kwargs):
        fn(self, *args, **kwargs)

    def _get_channel_to(self, chare_idx):
        if chare_idx not in self._channels_map:
            self._channels_map[chare_idx] = Channel(self,
                                                remote = self._comm[(chare_idx,)]
                                                )
            self._channels.append(self._channels_map[chare_idx])
        return self._channels_map[chare_idx]

Communicator.register(CharmCommunicator)

Request = None
