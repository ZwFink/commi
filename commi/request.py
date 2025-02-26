# from mpi4py import MPI
# Request = MPI.Request

# class Request:
#     @classmethod
#     def waitall(requests):
#         MPI.Request.Waitall(requests)
from charm4py import Chare, coro, Channel, charm, Reducer, Future
from collections import defaultdict
import numpy as np

def Waitsome(requests):
    # is there a better way to force a check for messages?
    charm.sleep(0)
    idxes = []
    for idx, r in enumerate(requests):
        value = r.recvMgr.tryReceiveFromChannelWithTag(r.ch, r.tag)
        if value is not None:
            idxes.append(idx)
            r._CopyBuf(value)
            r._Complete()
    if len(idxes) == 0:
        return None
    return idxes

def Waitall(requests):
    idxes = []
    for idx, r in enumerate(requests):
        r.Wait()
        idxes.append(idx)
    return idxes


INCOMPLETE=0
COMPLETE=1

class Request:
    def __init__(self):
        self.state = INCOMPLETE
    def Wait(self):
        return
    def _Complete(self):
        self.state = COMPLETE
    def Test(self):
        return self.state == COMPLETE

class RecvRequest(Request):
    def __init__(self, buf, recvMgr, ch, tag):
        super().__init__()
        self.buf = buf
        self.ch = ch
        self.recvMgr = recvMgr
        self.tag = tag

    def Wait(self):
        # TODO: tag
        d = self.recvMgr.receiveFromChannelWithTag(self.ch, self.tag)
        self._Complete()
        # print(tag, d)
        if self.buf is not None:
            np.copyto(self.buf, d)
        else:
            return d
    def _CopyBuf(self, value):
        np.copyto(self.buf, value)

class SendRequest(Request):
    # Currently, charm4py has no ZC
    def Wait(self):
        self._Complete()
        return

class RecvManager:
    def __init__(self):

        def __create():
            return defaultdict(list)
        self.whatever : defaultdict[Channel, defaultdict[int, list]] = defaultdict(__create)


    def tryReceiveFromChannelWithTag(self, ch: Channel, tag: int):
        sender = self.whatever[ch]
        retval = None
        if sender[tag]:
            retval = sender[tag].pop()
        while ch.ready():
            recvtag, msg = ch.recv()
            sender[recvtag].append(msg)
            if retval is None and recvtag == tag:
                retval = sender[recvtag].pop()
        return retval

    def receiveFromChannelWithTag(self, ch: Channel, tag: int):
        sender = self.whatever[ch]

        if tag == -1:
            try:
                truetag, received = next(iter(sender.items()))
            except StopIteration:
                truetag = tag
                received = 0
        else:
            received = len(sender[tag])
            truetag = tag
        if not received:
          recvtag, msg = ch.recv()
          sender[recvtag].append(msg)

          if tag == -1:
                truetag = recvtag
          else:
            while recvtag != tag:
              recvtag, msg = ch.recv()
              truetag = recvtag
              sender[recvtag].append(msg)

        # truetag != tag iff tag == -1
        return sender[truetag].pop(0)

