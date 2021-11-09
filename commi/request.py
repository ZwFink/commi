from mpi4py import MPI
Request = MPI.Request

class Request:
    @classmethod
    def waitall(requests):
        MPI.Request.Waitall(requests)
