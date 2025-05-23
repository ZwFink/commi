# from mpi4py import MPI
# Request = MPI.Request

# class Request:
#     @classmethod
#     def waitall(requests):
#         MPI.Request.Waitall(requests)
from charm4py import Chare, coro, Channel, charm, Reducer, Future
from collections import defaultdict
from .status import Status
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

def Waitany(requests, status=None):
    if status is None:
        status = Status()
    chs = list()
    for r in requests:
        r.ch._tag = r.tag
        r.ch._request = r
        chs.append(r.ch)
    for idx, channel in enumerate(charm.iwait(chs)):
        r = channel._request
        r.Wait()
        status.tag = channel._tag
        status.source = channel._chare_idx
        return idx
    return -1

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

    def __getstate__(self):
        return self.buf, None, None, self.tag

    def __setstate__(self, state):
        self.buf, self.ch, self.recvMgr, self.tag = state
        self.recvMgr = RecvManager()

class SendRequest(Request):
    # Currently, charm4py has no ZC
    def Wait(self):
        self._Complete()
        return

class RecvManager:
    def __init__(self):
        create = lambda: defaultdict(list)
        self.whatever : defaultdict[Channel, defaultdict[int, list]] = defaultdict(create)


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


    def __getstate__(self):
        state = self.__dict__.copy()
        del state['whatever']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.whatever = defaultdict(lambda: defaultdict(list))
