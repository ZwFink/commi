from charm4py import Chare, coro, Channel, charm
from . import Communicator
from typing import Any
class CharmCommunicator(Chare):
    def __init__(self, n_elems):
        self._comm = self.thisProxy
        self._size = n_elems
        self._this_index = self.thisIndex[0]

        self._channels_map = dict()
        self._channels = list()

        # TODO: How can we do wildcard recvs without this?
        for i in range(n_elems):
            if i != self.Get_rank():
                self._get_channel_to(i)

    def Send(self, buf: Any = None, dest: int = -1, tag: int = -1, status: Any = None):
        ch = self._get_channel_to(dest)
        ch.send(buf)
    def Recv(self, buf: Any = None, source: int = -1, tag: int = -1, status: Any = None):
        if source == -1:
            gen = charm.iwait(self._channels)
            recv = next(gen).recv()
            del gen
            return recv
        ch = self._get_channel_to(source)
        return ch.recv()

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
