from .communicator import Communicator, Request
from .status import Status, COMMI_STATUS_IGNORE, COMMI_STATUS_IGNORE
from .request import Request
from .mpi_communicator import MPICommunicator
from .charm_communicator import CharmCommunicator

def CreateCharmCommunicator(args, nelems):
    from charm4py import Array
    return Array(CharmCommunicator, nelems, args=args)

def Start(fn):
    from charm4py import charm
    charm.start(fn, classes=[CharmCommunicator])
