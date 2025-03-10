from .communicator import Communicator, Request
from .status import Status, COMMI_STATUS_IGNORE, COMMI_STATUS_IGNORE
from .request import Request, SendRequest, RecvRequest, RecvManager
#from .mpi_communicator import MPICommunicator
from .charm_communicator import *
from enum import Enum
from functools import wraps

from charm4py import charm

COMM_WORLD = None

class CommiBackendSpecifier(Enum):
    CHARM_BACKEND = 1
    MPI_BACKEND = 1
def CreateCharmCommunicator(args, nelems):
    from charm4py import Array
    print(f"Creating array with {nelems} elems")
    return Array(CharmCommunicator, nelems, args=args, useAtSync=True)

def Start(fn):
    from charm4py import charm
    charm.start(fn, classes=[CharmCommunicator])

def exit():
  charm.exit()


def commi_entry_point(backend_spec: CommiBackendSpecifier, num_pes: int):
    import __main__
    global COMM_WORLD
    if backend_spec == CommiBackendSpecifier.CHARM_BACKEND:
        def actual_decorator(func):
            @wraps(func)
            def wrapped(args):
                comm = CreateCharmCommunicator([num_pes], num_pes)
                COMM_WORLD = comm

                setattr(__main__, 'entry_point', func)
                charm.updateGlobals('entry_point', func, awaitable = True).get()
                comm.begin_exec(func)
            Start(wrapped)
        return actual_decorator
    elif backend_spec == CommiBackendSpecifier.MPI_BACKEND:
        pass
