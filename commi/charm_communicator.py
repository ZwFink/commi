from charm4py import Chare, coro, Channel, charm, Reducer, Future, register, Reducer
from . import Communicator
from . import Request, SendRequest, RecvRequest, RecvManager
from typing import Any
import numpy as np
import mpi4py
mpi4py.rc.initialize = False
from mpi4py import MPI

def reducer_map(op):
  if op == MPI.SUM:
    return  Reducer.sum
  if op == MPI.MAX:
    return Reducer.max
  if op == MPI.MIN:
    return Reducer.min
  if op == MPI.LOR:
    return Reducer.logical_or
  if op == MPI.LAND:
    return Reducer.logical_and
  print("We didn't find it!")
  return op


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

        # # TODO: How can we do wildcard recvs without this?
        # for i in range(n_elems):
            # if i != self.Get_rank():
                # self._get_channel_to(i)

    def __getstate__(self):
      state = self.__dict__.copy()
      parent_state = super().__getstate__()
      del state['_mgr']
      try:
        del state['_allgather_fut']
      except:
        pass
      return (parent_state, state)

    def __setstate__(self, state):
      parent_state, state = state
      super().__setstate__(parent_state)
      self.__dict__.update(state)
      self._mgr = RecvManager()
      self._allgather_fut = None

    @coro
    def Send(self, buf: Any = None, dest: int = -1, tag: int = -1, status: Any = None):
        ch = self._get_channel_to(dest)
        ch.send(tag, buf)
    @coro
    def Recv(self, buf: Any = None, source: int = -1, tag: int = -1, status: Any = None):
        if source == -1:
            while True:
              found = False
              for ch in self._channels:
                if ch.ready():
                  tag, recv = ch.recv()
                  recv_ch = ch
                  found = True
                  break 
              if found:
                break
              else:
                f = Future()
                f(0)
                f.get()
            #gen = charm.iwait(self._channels)
            #for ch in charm.iwait(self._channels):
            #  tag, recv = ch.recv()
            #  recv_ch = ch
            #  break
            if status:
               status.source = recv_ch._chare_idx
               status.tag = tag
            return recv
        recv = self._mgr.receiveFromChannelWithTag(self._get_channel_to(source), tag)
        # ch = self._get_channel_to(source)
        # tag, recv = ch.recv()
        return recv

    @coro
    def recv(self, buf: Any = None, source: int = -1, tag: int = -1, status: Any = None):
      return self.Recv(buf, source, tag, status)

    def Irecv(self, buf: Any = None, source: int = -1, tag: int = -1, status: Any = None):
        return RecvRequest(buf, self._mgr, self._get_channel_to(source), tag)

    def Isend(self, buf: Any = None, dest: int = -1, tag: int = -1, status: Any = None):
        self.Send(buf, dest, tag, status)
        return SendRequest()
    def isend(self, buf: Any = None, dest: int = -1, tag: int = -1, status: Any = None):
      return self.Isend(buf, dest, tag, status)

    @coro
    def barrier(self):
        self.allreduce().get()
    @coro
    def Barrier(self):
      self.barrier()

    @coro
    def owlreduce(self, data, op=None):
      real_op = reducer_map(op)
      was_scalar = False
      if np.isscalar(data):
        was_scalar = True
        data = np.array([data], dtype=data.dtype) 
      retval = self.allreduce(data, reducer=real_op).get()
      if was_scalar:
        return np.array(retval)
      if isinstance(data, np.ndarray):
        return np.array([retval])
      return retval

    @coro
    def bcast(self, data, root=0):
      self._bcast_fut = Future()
      if self.Get_rank() == root:
        self.thisProxy._bcast_recv(data, awaitable=True).get()
      result = self._bcast_fut.get()
      del self._bcast_fut
      return result

    def iprobe(self):
      return False
      

    def _bcast_recv(self, data):
      assert self._bcast_fut is not None
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
        return my_data[0]
            
      else:
        data = self._get_channel_to(root).recv()
        return data[0]

    @coro
    def scan(self, data):
      if self.Get_rank() == 0:
         retval = data
         self.Send(data, dest = 1)
         self.barrier()
         return retval

      if self.Get_rank() == self.Get_size() - 1:
        self.barrier()
        return self.Recv(source = self.Get_rank() - 1)
      
      target = self.Get_rank() + 1
      recv = self.Recv(self.Get_rank() - 1)
      retval = recv
      sendval = data + recv
      self.Send(sendval, dest = target)
      self.barrier()
      return retval 

    def Get_rank(self):
        return self.thisIndex[0]
    def Get_size(self):
        return self._size

    def get_communicator(self):
        return self._comm

    def Dup(self):
        # NOTE: This is dangerous
        return self

    def Free(self):
        pass

    @coro
    def redux(self, data, op, root=-1):
      real_op = reducer_map(op)
      if self.rank == root:
        self._allgather_fut = Future()
      self.reduce(self.thisProxy[0]._return_from_allgather, data, real_op)
      if self.rank == root:
        self._allgather_fut.get()
        return self._allgather_result

    @coro
    def allgather(self, sendobj: Any):
        # TODO: Expose LocalFuture to user code
        self._allgather_fut = Future()
        self.reduce(self.thisProxy._return_from_allgather, sendobj, Reducer.gather)
        self._allgather_fut.get()
        return self._allgather_result

    @coro
    def gather(self, sendobj: Any, root=-1):
      if self.rank == root:
        self._allgather_fut = Future()
      self.reduce(self.thisProxy[root]._return_from_allgather, sendobj, Reducer.gather)
      if self.rank == root:
        self._allgather_fut.get()
        return self._allgather_result

    def _return_from_allgather(self, result):
        self._allgather_result = result
        self._allgather_fut()
    @coro
    def begin_exec(self, fn, *args, **kwargs):
        fn(self, *args, **kwargs)

    def Migrate(self):
        self.AtSyncAndWait()

    @coro
    def _get_channel_to(self, chare_idx):
        if chare_idx not in self._channels_map:
            self._channels_map[chare_idx] = Channel(self,
                                                    remote=self._comm[(
                                                        chare_idx,)]
                                                    )
            self.thisProxy[chare_idx]._receive_channel_request(self.thisIndex[0], self.thisProxy[self.thisIndex[0]], awaitable=True).get()
            # self._channels_map[chare_idx]._chare_idx = chare_idx
            # self._channels.append(self._channels_map[chare_idx])
        return self._channels_map[chare_idx]

    def _receive_channel_request(self, remote_idx, remote_chare):
      if remote_idx not in self._channels_map:
        self._channels_map[remote_idx] = Channel(self, remote=remote_chare)
        self._channels_map[remote_idx]._chare_idx = remote_idx
        self._channels.append(self._channels_map[remote_idx])


Communicator.register(CharmCommunicator)

Request = None

