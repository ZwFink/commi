from charm4py import Chare, coro, Channel, charm, Reducer, Future, register, noproxy
from . import Communicator
from . import Request, SendRequest, RecvRequest, RecvManager
from typing import Any
import numpy as np



@register
class CharmCommunicator(Chare):
    def __init__(self, n_elems):
        self._mgr = RecvManager()
        self._comm = self.thisProxy
        self._size = n_elems
        self._this_index = self.thisIndex[0]
        self.rank = self._this_index

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
        recv = self._mgr.receiveFromChannelWithTag(self._get_channel_to(source), tag)
        # ch = self._get_channel_to(source)
        # tag, recv = ch.recv()
        return recv

    def recv(self, buf: Any = None, source: int = -1, tag: int = -1, status: Any = None):
      return self.Recv(buf, source, tag, status)

    def Irecv(self, buf: Any = None, source: int = -1, tag: int = -1, status: Any = None):
        return RecvRequest(buf, self._mgr, self._get_channel_to(source), tag)

    def Isend(self, buf: Any = None, dest: int = -1, tag: int = -1, status: Any = None):
        if tag == 0:
            print("Sending a with tag 0!")
        self.Send(buf, dest, tag, status)
        return SendRequest()
    def isend(self, buf: Any = None, dest: int = -1, tag: int = -1, status: Any = None):
      return self.Isend(buf, dest, tag, status)

    def barrier(self):
        self.allreduce().get()
    def Barrier(self):
      self.barrier()

    @coro
    def bcast(self, data, root=0):
      self._bcast_fut = Future()
      if self.Get_rank() == root:
        self.thisProxy._bcast_recv(data, awaitable=True).get()
      return self._bcast_fut.get()

    def iprobe(self):
      return False
      

    def _bcast_recv(self, data):
      self._bcast_fut(data)

    @coro
    def scatter(self, data, root=0):
      if self.Get_rank() == root:
        p = self.Get_size()
        my_data = None
        n = len(data)
        sublist_size = n // p

        sublists = []
        for i in range(p):
            start_index = i * sublist_size
            if i == p - 1:  # Assign the remaining elements to the last processor
                end_index = n
            else:
                end_index = (i + 1) * sublist_size
            sublists.append(data[start_index:end_index])
        for chare in range(0, self.Get_size()):
          chare_data = sublists[chare]
          if chare == self.Get_rank():
            my_data = chare_data
          else:
            self._get_channel_to(chare).send(chare_data)
        return my_data
            
      else:
        data = self._get_channel_to(root).recv()
        return data

    def Get_rank(self):
        print(f"This rank has index: {self.thisIndex}")
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
        print("Args:", args, "Kwargs:", kwargs)
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
